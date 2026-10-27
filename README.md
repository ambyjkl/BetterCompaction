# Better Compaction

Context compaction for [pi](https://pi.dev) that reuses the backend's prompt
cache: instead of summarizing the conversation in a throwaway chat, it
appends the compaction instruction to the conversation itself, so the model
re-reads almost nothing.

## Why

When an AI agent (or an AI harness) needs to compact a conversation, it
usually does this:

1. Take the entire conversation.
2. Start a **new** chat with a system prompt explaining how to compact it.
3. Put the entire conversation below that prompt and ask the model to
   produce the summary.

That works, but it has two problems.

**It throws the cache away.** The summarization chat shares nothing with the
chat that was just running, so the backend's KV/prompt cache is useless: the
model has to pre-fill the *entire* conversation from scratch just to produce
a summary. And compaction happens precisely when the conversation is already
very long — so that full re-read takes a long time, every single time.

**The instruction sits at the top.** Some research has found that, for some
models, an instruction placed at the *end* of a long context performs better
than one placed at the beginning.

Better Compaction flips the design: the summarization request is the
conversation the agent was *already having* — truncated at the compaction cut
point, sent as real messages — with the compaction instruction appended as
the **final message**. The prefix is byte-identical to what the backend
already computed and cached, so it is reused and only the short appended
instruction has to be pre-filled. (On one ~94k-token session, the
summarization request reported 94.4k tokens served from cache.)

The "kept tail" — the most recent messages that pi retains verbatim after
compaction — is deliberately not sent, so the summary doesn't re-describe
messages that remain in the session.

**Anthropic is the exception.** It reads cache entries only at positions an
earlier request wrote (its explicit breakpoints, found by a 20-block
lookback), not at an arbitrary cut. A request truncated mid-conversation reads
only the system prompt and tools, which showed up as ~4k cache-read tokens out
of ~125k. On `anthropic-messages` models the request therefore keeps the whole
conversation, ending exactly where the agent's last request wrote its cache
entry, and the appended instruction names the boundary and tells the model to
summarize only what comes before it.

Request-only system-prompt overrides are mirrored too. For example,
`pi-smart-timeout` returns a replacement prompt from `before_agent_start`;
Pi collapses the prompt and current tools into one request header without
persisting that replacement. Better Compaction observes those run options and
applies the same projection, including the full current tool state even when
some tool updates are in the kept tail.

**After loading/reloading or switching sessions, send a normal message before
compacting** if you use request-only prompt overrides. Their previous values
are not recoverable from the session file. The extension warns when it has not
observed a run and uses the stored transcript rather than guessing an override.
Compaction details record the projection as `forced-system-prompt`, `transcript`,
or `unobserved` for diagnostics.

Repeated compactions reuse the checkpoint already in the cached prefix instead
of copying it into the final instruction again. The model rewrites a concise
operational checkpoint rather than growing a historical log.

The summary format, bookkeeping, file tracking, events, and kept-messages
rebuild follow pi's normal compaction lifecycle. The extension only supplies the summary through the
`session_before_compact` event; pi does the rest with its normal path.

## Benchmarks

Better Compaction was developed with **Qwen3.8-27B** in mind (local
llama.cpp deployment), but it works with any model and any backend that
supports prefix/KV caching.

There are no formal benchmarks — compaction time depends heavily on your
hardware and your model. On my machine, with Qwen3.8-27B, the same session
went from **about 13–15 minutes** of compaction down to **about 6.5 minutes**,
and to **about 2 minutes** with compaction steered to skip thinking
(`/bc thinking off`).

## Install

### Requirements

The extension requires **pi** `0.87.1` or [higher](https://github.com/earendil-works/pi/releases).

### Option A — single file (simplest)

Copy the extension file into your pi extensions directory:

```bash
cp extensions/better-compaction.ts ~/.pi/agent/extensions/
```

Restart pi (or start a new session). Done.

### Option B — as a pi package (local path)

```bash
pi install /absolute/path/to/BetterCompaction
```

### Option C — as a pi package (git)

```bash
pi install git:github.com/NotSoSeriouss/BetterCompaction
pi update --extensions   # reconcile to the pinned ref later
```

## Usage

### The `/bc` command

| Command | Effect |
| --- | --- |
| `/bc` | Show effective options: enabled or not, summary budget, compaction thinking level, session thinking level, and which settings file provides the config |
| `/bc on` | Enable the extension (default) |
| `/bc off` | Disable the extension — pi's built-in compaction takes over |
| `/bc thinking off\|inherit` | Select minimal/no reasoning for supported OpenAI models, low effort for supported Anthropic models, or prompt-only steering elsewhere; `/bc` shows which mechanism applies |
| `/bc report on\|off` | Show / hide the post-compaction report message (timing and cache read) |

`/bc` writes to the settings file that **currently provides** the
`betterCompaction` section — the project file (`.pi/settings.json`) if it
defines one, otherwise the global file (`~/.pi/agent/settings.json`) — so it
always edits the file that is actually in effect. Changes are read at
compaction time, so they apply at the **next compaction** without restarting.

### Settings

`~/.pi/agent/settings.json` (global) or `.pi/settings.json` (project, wins):

```json
{
  "betterCompaction": {
    "enabled": true,
    "report": true,
    "summaryMaxTokens": 8192,
    "thinking": "inherit"
  }
}
```

| Option | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | `false` → the extension stays silent and pi's built-in compaction runs instead |
| `report` | `true` | `false` → don't append the "Compaction completed in Xs ..." message to the chat after a compaction (toggle with `/bc report off`) |
| `thinking` | `"inherit"` | `"inherit"` keeps the session level and asks for brief thinking. `"off"` uses an append-only effort update on supported native OpenAI models (`none`, or the lowest supported effort) and Anthropic models (`low`). Other models use prompt-only steering. The top-level thinking/effort never changes — see below |
| `summaryMaxTokens` | `8192` | Requested generation budget; integer of at least `1024`. The prompt targets half the budget for summary text (default **4096 tokens**), reduced for smaller model limits. Provider limitations apply — see below |

### Summary size and repeated compaction

The appended instruction asks for a complete checkpoint within a **4096-token
text target** by default. It prioritizes current goals, constraints, active work,
blockers, and next steps; completed work is consolidated into a few outcomes.
Old experiment logs and artifact catalogs should be referenced through their
on-disk documents, not reproduced. An oversized previous checkpoint is condensed,
not treated as a minimum length.

The previous checkpoint stays untouched in the cached conversation prefix. It is
not copied into the appended instruction a second time. A fallback copy is used
only when the preparation supplies a checkpoint that is genuinely absent from
the projection; explicitly omitted checkpoints are never restored.

`summaryMaxTokens` is **not a universal hard limit**. OpenAI Responses and adaptive
Anthropic requests receive the requested output budget, capped by the model.
Codex's adapter does **not** send a wire output-token cap, so it relies on the
appended text target. For budget-based thinking adapters, the existing generation
ceiling is retained: shrinking it can change a rendered thinking budget and lose
the prefix cache. Providers may also count or reserve reasoning tokens differently.
The text target does not grow just because `compaction.reserveTokens` is large.
Increase `summaryMaxTokens` in settings if your checkpoint genuinely needs more
space; `/bc` shows the configured budget.

A token-capped response is rejected rather than saved as a partial checkpoint,
even if it contains tool calls. File-operation tracking is appended by the
extension, outside the model's text budget, and merged from its previous
checkpoint metadata; the model is told not to generate duplicate tracking XML.
Compaction details record `summaryBudget` and `previousSummarySource` for diagnostics.
No existing session or checkpoint is rewritten in place.

### How thinking is handled without rewriting the cached prefix

Changing top-level thinking settings can rewrite a model's rendered prompt
and lose the prefix cache. This affects local chat templates, **OpenAI's hidden
instructions, and Anthropic's rendered thinking configuration**. Better
Compaction therefore always keeps the session's
top-level reasoning setting, even when the session is at `max`.

- `"inherit"` (default) keeps that effort and appends a brief-thinking prompt.
- `"off"` on native **OpenAI GPT-6-family Responses/Codex models** appends a
  `configuration_update` immediately before the summary instruction. Its effort
  is `none` where supported, otherwise the lowest level allowed by Pi's model
  metadata, mapped to the provider's wire value (for example, `low` on a model
  that cannot disable reasoning). The earlier input items and top-level effort
  are unchanged.
- `"off"` on native **Anthropic Messages models** with
  `compat.supportsMidConvoEffort: true` requests **`low` per-message effort**.
  Pi already appends an effort-only system message after the summary user turn;
  only that final marker is changed. Historical markers, signed thinking blocks,
  cache controls, and top-level thinking/effort stay unchanged. **`low` does not
  guarantee zero thinking.** Already-disabled sessions are left alone; models
  that cannot disable thinking can still lower effort in their adaptive mode.
- `"off"` on other models/backends remains **best-effort prompt-only steering**:
  it asks the model to skip thinking, but does not disable API reasoning.

For example, a `max` session with `/bc thinking off` on GPT-6.1 Terra sends
this item just before the summary prompt, while top-level effort stays `max`:

```json
{ "type": "configuration_update", "reasoning": { "effort": "none" } }
```

On supported Anthropic models, the final summary turn instead ends with:

```json
{ "role": "system", "content": [], "output_config": { "effort": "low" } }
```

This requires adaptive thinking and the
`mid-conversation-output-config-2026-07-01` beta, which Pi enables for models
with the verified compatibility flag. The extension validates the existing
mode, beta, and terminal marker rather than changing top-level settings or
silently guessing support. Custom providers, Bedrock, and models without the
flag retain prompt-only steering. Compaction details record the selected
`reasoningUpdate` for diagnostics.

`/bc` and `/bc thinking off` show the selected mechanism and effort, or explicitly
say `prompt only; API thinking unchanged`. Switch back with `/bc thinking inherit`.
The update exists only in the summarization request: it does not lower the main
agent's effort or persist in the session history. Unsupported-update API errors
are surfaced; the extension does not silently retry with changed top-level effort.

This uses Pi's existing `onPayload` hook after provider wire conversion, before
the request is sent. No Pi patch is needed. Server acceptance and cache reuse
still need to be verified for the selected endpoint, especially custom gateways.
OpenAI compatibility is restricted to the native `openai` and `openai-codex`
providers with GPT-6-family IDs and Responses APIs; Anthropic compatibility is
restricted to the native `anthropic` provider and its verified per-message effort
flag. No model-name heuristic enables Anthropic support.

See OpenAI's [reasoning-update documentation](https://developers.openai.com/api/docs/guides/reasoning#change-reasoning-mid-conversation)
and [cache-affecting settings](https://developers.openai.com/api/docs/guides/prompt-caching).
Their restrictions on server-side automatic compaction/truncation do not refer
to Pi's local compaction; this extension generates a summary through an ordinary
Responses request, not `/responses/compact`.
For Anthropic, see [per-message effort](https://platform.claude.com/docs/en/build-with-claude/effort#change-effort-mid-conversation)
and [cache-affecting settings](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#what-invalidates-the-cache).

### Notifications

- **After each successful compaction** (disable with `/bc report off`):
  `Compaction completed in 42.3s (35.1s thinking, 7.2s generating, 94.4k tokens from cache).`
  The thinking breakdown is shown only when the model actually thought; the
  cache figure shows how many tokens the prefix cache served. The message is
  appended to the chat on pi's `session_compact` event — which fires only
  after pi has applied the compaction result and finished the compaction
  lifecycle — with the timing measured on the final summarization attempt. It
  is stored as a custom session entry, so it stays in the transcript (and
  survives session reloads) but is **never sent to the LLM**. It is *not*
  appended for pi's built-in compaction (the report is stashed per session
  during `session_before_compact` and only exists when this extension
  supplied the summary; a failed compaction discards the stashed report).
- **Once per session, when no-thinking compaction is requested:** reports the
  append-only effort update, or warns that the model only has prompt-level
  steering. Some models summarize worse without reasoning.

The extension also mirrors the settings pi applies to normal agent requests
(`images.blockImages`, `retry.*`, provider timeouts, transport, thinking
budgets), reading the same settings files pi merges, so the summarization
request stays identical to what the agent sends — including its thinking mode,
which is the key to reusing the prompt cache (see above).

## Repository layout

```
better-compaction/
├── package.json               # pi package manifest (pi.extensions, pi-package keyword)
├── README.md
├── .gitignore
└── extensions/
    └── better-compaction.ts   # the extension (single file, no build step)
```

- `package.json` declares `"pi": { "extensions": ["./extensions"] }` and the
  `pi-package` keyword, so `pi install` (local path or git) loads
  `extensions/*.ts` automatically.
- The pi runtime packages (`@earendil-works/pi-coding-agent`,
  `@earendil-works/pi-ai`) are `peerDependencies` with `*` ranges: they are
  provided by pi itself and are never bundled.
- There are no other dependencies and no build step — the extension is plain
  TypeScript loaded by pi's jiti loader.

## Development notes

- Type-checking locally: pi resolves the peer packages with its own loader
  aliases at runtime. To type-check outside pi, point a `tsconfig.json`
  `paths` entry for `@earendil-works/pi-coding-agent` and
  `@earendil-works/pi-ai` at the installed pi package
  (`<pi-install>/dist/index.d.ts` and
  `<pi-install>/node_modules/@earendil-works/pi-ai/dist/index.d.ts`).
- Known limitations (documented, intentional):
  - Forced prompts returned by `before_agent_start` are mirrored once a run
    has been observed. Arbitrary `context`/`context_with_system` transformations,
    hidden tool declarations, or provider-payload rewrites are not replayed;
    they can still make the rebuilt prefix differ and reduce or eliminate cache
    reuse. A cache hit is not guaranteed.
  - Provider attribution headers (telemetry) that pi adds for a few specific
    providers are not replicated; they don't affect prompt caching.
  - Pi does not merge file tracking from arbitrary extension-generated
    checkpoints. Better Compaction carries its own checkpoint metadata forward;
    other extensions' custom tracking formats are not interpreted.

## License

MIT
