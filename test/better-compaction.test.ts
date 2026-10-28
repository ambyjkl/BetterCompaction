/**
 * better-compaction report lifecycle tests.
 *
 * Regression guard for the post-compaction report: it must be appended as a
 * custom session entry (persistent in the chat, never sent to the LLM) on pi's
 * `session_compact` event (which fires only after pi has applied the
 * compaction result), NOT inside `session_before_compact` — and only for
 * compactions this extension actually produced.
 *
 * Run with: bun test
 */
/// <reference types="bun-types" />
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync, readFileSync, copyFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { buildSessionProjection, convertToLlm, type SessionEntry } from "@earendil-works/pi-coding-agent";
import { getCurrentSystemMessage, getCurrentSystemPrompt, normalizeContext } from "@earendil-works/pi-ai";
import type { Api, AssistantMessage, Model, SimpleStreamOptions, TranscriptContext } from "@earendil-works/pi-ai";
import extension from "../extensions/better-compaction";

// ---------------------------------------------------------------------------
// Mock harness
// ---------------------------------------------------------------------------

interface ApiMock {
	commands: Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>;
	/** event → handler (in registration order) */
	handlers: Map<string, unknown[]>;
	/** entries appended via pi.appendEntry, in order */
	entries: Array<{ customType: string; data?: unknown }>;
	on: (event: string, handler: unknown) => () => void;
	registerCommand: (name: string, def: unknown) => void;
	registerEntryRenderer: (customType: string, renderer: unknown) => void;
	appendEntry: (customType: string, data?: unknown) => void;
}

function createApiMock(): ApiMock {
	const handlers = new Map<string, unknown[]>();
	const commands: ApiMock["commands"] = new Map();
	const entries: Array<{ customType: string; data?: unknown }> = [];
	return {
		handlers,
		commands,
		entries,
		on: (event, handler) => {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
			return () => {
				const i = list.indexOf(handler);
				if (i >= 0) list.splice(i, 1);
			};
		},
		registerCommand: (name, def) => {
			commands.set(name, def as { handler: (args: string, ctx: unknown) => Promise<void> });
		},
		registerEntryRenderer: () => {},
		appendEntry: (customType, data) => {
			entries.push({ customType, data });
		},
	};
}

/** Fire all handlers in registration order, retaining the last non-void result. */
async function fire(api: ApiMock, event: string, eventPayload: unknown, ctx: unknown) {
	const list = api.handlers.get(event);
	expect(list, `no handler registered for ${event}`).toBeDefined();
	let result: unknown;
	for (const handler of list! as Array<(e: unknown, c: unknown) => unknown>) {
		const value = await handler(eventPayload, ctx);
		if (value !== undefined) result = value;
	}
	return result;
}

// ---------------------------------------------------------------------------
// Session entry / stream fixtures
// ---------------------------------------------------------------------------

function makeEntries(): { entries: unknown[]; leafId: string; keptEntryId: string } {
	const mk = (i: number, message: unknown) => ({
		type: "message",
		id: `entry-${i}`,
		parentId: i === 0 ? null : `entry-${i - 1}`,
		timestamp: new Date(1_700_000_000_000 + i * 1000).toISOString(),
		message,
	});
	const user = (text: string) => ({
		role: "user" as const,
		content: text,
		timestamp: Date.now(),
	});
	const assistant = (text: string) => ({
		role: "assistant" as const,
		content: [{ type: "text" as const, text }],
		api: "openai-completions" as const,
		provider: "test",
		model: "test-model",
		usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0 },
		stopReason: "stop" as const,
		timestamp: Date.now(),
	});
	const entries = [
		mk(0, user("Read the codebase and tell me what it does.")),
		mk(1, assistant("It is a small CLI tool.")),
		mk(2, user("Now add a config file option.")), // firstKeptEntryId — cut point
		mk(3, assistant("Done, added --config.")),
		mk(4, user("Great, commit it.")),
	];
	return { entries, leafId: "entry-4", keptEntryId: "entry-2" };
}

/**
 * A session whose model view diverges from the raw session file: one message
 * was OMITTED from the model's context (e.g. context-overflow recovery) and
 * one assistant message was REPLACED, both via context_edit entries before
 * the cut point. Issue #3 regression fixture.
 */
function makeEditedEntries(): { entries: unknown[]; leafId: string; keptEntryId: string } {
	const mk = (i: number, entry: Record<string, unknown>) => ({
		...entry,
		id: `entry-${i}`,
		parentId: i === 0 ? null : `entry-${i - 1}`,
		timestamp: new Date(1_700_000_000_000 + i * 1000).toISOString(),
	});
	const user = (text: string) => ({ role: "user" as const, content: text, timestamp: Date.now() });
	const assistant = (text: string) => ({
		role: "assistant" as const,
		content: [{ type: "text" as const, text }],
		api: "openai-completions" as const,
		provider: "test",
		model: "test-model",
		usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0 },
		stopReason: "stop" as const,
		timestamp: Date.now(),
	});
	return {
		entries: [
			mk(0, { type: "message", message: user("Read the codebase and tell me what it does.") }),
			mk(1, { type: "message", message: assistant("It is a small CLI tool.") }),
			mk(2, { type: "message", message: user("Original omitted question.") }),
			mk(3, {
				type: "message",
				message: assistant("Original long answer the model no longer sees."),
			}),
			mk(4, { type: "context_edit", targetId: "entry-2", replacement: null }),
			mk(5, { type: "context_edit", targetId: "entry-3", replacement: { content: "Shortened answer." } }),
			mk(6, { type: "message", message: user("Now add a config file option.") }), // firstKeptEntryId
			mk(7, { type: "message", message: assistant("Done, added --config.") }),
		],
		leafId: "entry-7",
		keptEntryId: "entry-6",
	};
}

function makePreparation(keptEntryId: string) {
	return {
		firstKeptEntryId: keptEntryId,
		messagesToSummarize: [] as unknown[],
		turnPrefixMessages: [] as unknown[],
		isSplitTurn: false,
		tokensBefore: 123_456,
		fileOps: { read: new Set<string>(), written: new Set<string>(), edited: new Set<string>() },
		settings: { reserveTokens: 20_000 },
	};
}

function makeSummaryMessage(overrides?: {
	cacheRead?: number;
	stopReason?: "stop" | "error" | "length";
	errorMessage?: string;
	text?: string;
}) {
	return {
		role: "assistant" as const,
		content: [{ type: "text" as const, text: overrides?.text ?? "## Goal\nKeep the test green." }],
		api: "openai-completions" as const,
		provider: "test",
		model: "test-model",
		usage: {
			input: 100,
			output: 40,
			cacheRead: overrides?.cacheRead ?? 12_345,
			cacheWrite: 0,
		},
		stopReason: overrides?.stopReason ?? ("stop" as const),
		...(overrides?.errorMessage ? { errorMessage: overrides.errorMessage } : {}),
		timestamp: Date.now(),
	};
}

/** Fake LLM stream with a ~300ms thinking span so the report shows the split. */
function makeStream(message: unknown) {
	const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
	return {
		async *[Symbol.asyncIterator]() {
			yield { type: "thinking_start" };
			await sleep(300);
			yield { type: "thinking_end" };
			yield { type: "done", message };
		},
	};
}

/**
 * Fake LLM stream in the order the OpenAI-compatible adapter actually emits:
 * thinking_start … text_start (answer begins) … then thinking_end ONLY at
 * stream end (blocks are finalized in bulk after the SSE loop).
 */
function makeOpenAiOrderStream(message: unknown) {
	const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
	return {
		async *[Symbol.asyncIterator]() {
			yield { type: "thinking_start" };
			await sleep(300);
			yield { type: "text_start" };
			await sleep(600);
			yield { type: "thinking_end" }; // late: only at stream end
			yield { type: "done", message };
		},
	};
}

function makeCtx(opts: {
	sessionId: string;
	cwd: string;
	notify: (message: string, type?: string) => void;
	/** produce() results, consumed one per streamSimple call */
	messages: unknown[];
	hasUI?: boolean;
	/** stream shape used by the fake streamSimple (default: makeStream) */
	streamFactory?: (message: unknown) => unknown;
	entries?: unknown[];
	leafId?: string;
}) {
	const defaults = makeEntries();
	const entries = opts.entries ?? defaults.entries;
	const leafId = opts.leafId ?? defaults.leafId;
	const calls: unknown[][] = [];
	return {
		calls,
		model: {
			id: "test-model",
			provider: "test",
			maxTokens: 8192,
			reasoning: true,
		},
		modelRegistry: {
			streamSimple: async (...args: unknown[]) => {
				calls.push(args);
				const msg = opts.messages.shift() ?? makeSummaryMessage();
				return (opts.streamFactory ?? makeStream)(msg);
			},
		},
		sessionManager: {
			getEntries: () => entries,
			getLeafId: () => leafId,
			getSessionId: () => opts.sessionId,
		},
		cwd: opts.cwd,
		hasUI: opts.hasUI ?? true,
		thinkingLevel: "off",
		signal: new AbortController().signal,
		ui: { notify: opts.notify },
	};
}

function beforeCompactEvent(keptEntryId: string) {
	return {
		type: "session_before_compact",
		preparation: makePreparation(keptEntryId),
		branchEntries: [],
		reason: "manual",
		willRetry: false,
		signal: new AbortController().signal,
	};
}

function compactEvent() {
	return {
		type: "session_compact",
		compactionEntry: { type: "compaction", id: "c1", parentId: null, timestamp: "" },
		fromExtension: true,
		reason: "manual",
		willRetry: false,
	};
}

function compactFailedEvent() {
	return {
		type: "session_compact_failed",
		reason: "manual",
		errorMessage: "boom",
		aborted: false,
		willRetry: false,
		fromExtension: true,
	};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

let tmp: string;
let projectDir: string;
const originalAgentDir = process.env.PI_CODING_AGENT_DIR;

beforeAll(() => {
	const root = join(tmpdir(), process.env.PI_SESSION_ID ?? `better-compaction-${process.pid}`);
	mkdirSync(root, { recursive: true });
	tmp = mkdtempSync(join(root, "bc-test-"));
	// Hermetic pi agent dir (global settings): empty → all defaults.
	process.env.PI_CODING_AGENT_DIR = join(tmp, "agent");
	mkdirSync(process.env.PI_CODING_AGENT_DIR);
	projectDir = join(tmp, "project");
	mkdirSync(projectDir);
});

afterAll(() => {
	rmSync(tmp, { recursive: true, force: true });
	if (originalAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
	else process.env.PI_CODING_AGENT_DIR = originalAgentDir;
});

function freshExtension(): ApiMock {
	const api = createApiMock();
	extension(api as never);
	return api;
}

describe("single-file installation", () => {
	test("loads an isolated copy without any repository helper files", async () => {
		const cwd = join(tmp, "standalone");
		mkdirSync(cwd);
		const path = join(cwd, "better-compaction.ts");
		copyFileSync(join(__dirname, "../extensions/better-compaction.ts"), path);
		// Use Pi's real loader: it supplies runtime package aliases just as it
		// does when this single file is copied to ~/.pi/agent/extensions/.
		const agentDist = dirname(Bun.resolveSync("@earendil-works/pi-coding-agent", __dirname));
		const { loadExtensions } = await import(pathToFileURL(join(agentDist, "core/extensions/loader.js")).href);
		const result = await loadExtensions([path], cwd);
		expect(result.errors).toEqual([]);
		expect(result.extensions).toHaveLength(1);
	});
});

describe("request-only system prompt projection", () => {
	const tool = (name: string, description: string) => ({
		name, description, parameters: { type: "object", properties: {} },
	});
	const originalTool = tool("read", "Original schema");
	const currentTool = tool("read", "Updated schema");
	const addedTool = tool("new_tool", "Added in the kept tail");

	function setup() {
		const { entries } = makeEntries();
		const timestamp = 1_700_000_000_000;
		entries.unshift({
			type: "message", id: "system-0", parentId: null,
			timestamp: new Date(timestamp).toISOString(),
			message: {
				role: "system", content: "Original system prompt", timestamp,
				sections: { rules: "Original rules" },
				toolsAdded: [originalTool, tool("removed_tool", "Removed later")],
			},
		});
		(entries[1] as { parentId: string | null }).parentId = "system-0";
		entries.push({
			type: "message", id: "system-1", parentId: "entry-4",
			timestamp: new Date(timestamp + 1000).toISOString(),
			message: {
				role: "system", content: "", timestamp: timestamp + 1000,
				sections: { rules: "Updated rules" }, toolsAdded: [currentTool, addedTool],
				toolsRemoved: [{ name: "removed_tool" }],
			},
		});
		const messages = convertToLlm(buildSessionProjection(entries as SessionEntry[], "system-1").messages);
		const recordedPrompt = getCurrentSystemPrompt(messages);
		const api = freshExtension();
		const ctx = {
			...makeCtx({
				sessionId: "forced-prompt", cwd: projectDir, notify: () => {}, messages: [],
				entries, leafId: "system-1",
				streamFactory: (message) => ({ async *[Symbol.asyncIterator]() { yield { type: "done", message }; } }),
			}),
			getSystemPrompt: () => recordedPrompt,
		};
		return { api, ctx, messages, recordedPrompt };
	}

	test("mirrors late-handler overrides using the full current tools, without summarizing the kept tail", async () => {
		const { api, ctx, messages, recordedPrompt } = setup();
		const original = structuredClone(messages);
		const options: { forceSystemPrompt?: string } = {};
		await fire(api, "before_agent_start", { systemPromptOptions: options }, ctx);
		// Simulate smart-timeout running AFTER BetterCompaction's handler.
		options.forceSystemPrompt = `${recordedPrompt}\nTimeout policy`;
		// After settlement Pi exposes the BASE prompt, not the cached override.
		ctx.getSystemPrompt = () => recordedPrompt;
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(request.messages[0]).toMatchObject({
			role: "system", content: options.forceSystemPrompt, toolsAdded: [currentTool, addedTool],
		});
		expect(request.messages.filter((m) => m.role === "system")).toHaveLength(1);
		expect(request.messages.slice(1, -1)).toEqual(messages.slice(1, 3));
		expect(JSON.stringify(request.messages.slice(1, -1))).not.toContain("Now add a config file option");
		expect(messages).toEqual(original);
		expect(result).toHaveProperty("compaction.details.promptProjection", "forced-system-prompt");
	});

	test("a forced prompt equal to the rendered transcript still collapses system/tool deltas", async () => {
		const { api, ctx, recordedPrompt } = setup();
		await fire(api, "before_agent_start", { systemPromptOptions: { forceSystemPrompt: recordedPrompt } }, ctx);
		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(request.messages[0]).toMatchObject({ content: recordedPrompt, toolsAdded: [currentTool, addedTool] });
	});

	test("an empty forced prompt still collapses tool state", async () => {
		const { api, ctx } = setup();
		await fire(api, "before_agent_start", { systemPromptOptions: { forceSystemPrompt: "" } }, ctx);
		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(request.messages[0]).toMatchObject({ role: "system", content: "", toolsAdded: [currentTool, addedTool] });
	});

	test("session-start and shutdown discard previous request-only overrides", async () => {
		const { api, ctx, messages } = setup();
		for (const event of ["session_start", "session_shutdown"] as const) {
			await fire(api, "before_agent_start", { systemPromptOptions: { forceSystemPrompt: "Previous run" } }, ctx);
			await fire(api, event, {}, ctx);
			await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
			const request = ctx.calls.at(-1)![1] as TranscriptContext;
			expect(request.messages.slice(0, -1)).toEqual(messages.slice(0, 3));
		}
	});

	test("an override observed for another session cannot change this session", async () => {
		const { api, ctx, messages } = setup();
		const otherCtx = { ...ctx, sessionManager: { ...ctx.sessionManager, getSessionId: () => "other-session" } };
		await fire(api, "before_agent_start", { systemPromptOptions: { forceSystemPrompt: "Other session" } }, otherCtx);
		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(request.messages.slice(0, -1)).toEqual(messages.slice(0, 3));
	});

	test("after reload, warns rather than guessing a lost request-only override from the current prompt", async () => {
		const { api, ctx, messages } = setup();
		const notices: string[] = [];
		ctx.ui.notify = (message: string) => { notices.push(message); };
		ctx.getSystemPrompt = () => "A pending configuration change, not necessarily a forced request";
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(request.messages.slice(0, -1)).toEqual(messages.slice(0, 3));
		expect(notices.at(-1)).toContain("no normal run observed");
		expect(result).toHaveProperty("compaction.details.promptProjection", "unobserved");
	});

	test("does not collapse an observed unforced run even if the desired prompt later changes", async () => {
		const { api, ctx, messages } = setup();
		await fire(api, "before_agent_start", { systemPromptOptions: {} }, ctx);
		ctx.getSystemPrompt = () => "A pending configuration change, not a sent override";
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(request.messages.slice(0, -1)).toEqual(messages.slice(0, 3));
		expect(result).toHaveProperty("compaction.details.promptProjection", "transcript");
	});

	test("native Codex serialization preserves the full wire prefix with a forced prompt and low effort update", async () => {
		const { api, ctx, messages, recordedPrompt } = setup();
		const model: Model<"openai-codex-responses"> = {
			id: "gpt-6.1-sol", name: "Test Sol", api: "openai-codex-responses", provider: "openai-codex",
			baseUrl: "https://chatgpt.com/backend-api", reasoning: true, input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 500_000, maxTokens: 128_000,
			thinkingLevelMap: { off: null, minimal: "low", xhigh: "xhigh" },
			compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true, supportsToolSearch: true },
		};
		const solCtx = { ...ctx, model, thinkingLevel: "xhigh", getSystemPrompt: () => `${recordedPrompt}\nTimeout policy` };
		await fire(api, "before_agent_start", { systemPromptOptions: { forceSystemPrompt: solCtx.getSystemPrompt() } }, solCtx);
		await api.commands.get("bc")!.handler("thinking off", solCtx);
		try {
			await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), solCtx);
			const context = solCtx.calls[0]![1] as TranscriptContext;
			const options = solCtx.calls[0]![2] as SimpleStreamOptions;
			const current = getCurrentSystemMessage(messages)!;
			const mainContext = normalizeContext({ messages: [
				{ role: "system", content: solCtx.getSystemPrompt(), toolsAdded: current.toolsAdded!, timestamp: 0 },
				...messages.filter((m) => m.role !== "system"),
			] });
			const { streamSimple } = await import("@earendil-works/pi-ai/api/openai-codex-responses");
			const jwt = `test.${Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "test" } })).toString("base64url")}.test`;
			async function capture(request: TranscriptContext, inject: boolean) {
				let wire: unknown;
				const result = await streamSimple(model, request, {
					...options, apiKey: jwt,
					onPayload: async (payload, m) => {
						wire = inject ? await options.onPayload!(payload, m) : payload;
						throw new Error("captured-before-network");
					},
				}).result();
				expect(result.errorMessage).toContain("captured-before-network");
				return wire as { input: unknown[]; reasoning: { effort: string } };
			}
			const main = await capture(mainContext, false);
			const summary = await capture(context, true);
			expect(summary.reasoning.effort).toBe("xhigh");
			expect({ ...summary, input: main.input }).toEqual(main);
			expect(summary.input.slice(0, -2)).toEqual(main.input.slice(0, summary.input.length - 2));
			expect(summary.input.at(-2)).toEqual({ type: "configuration_update", reasoning: { effort: "low" } });
		} finally {
			await api.commands.get("bc")!.handler("thinking inherit", solCtx);
		}
	});
});

describe("bounded summaries and repeated compaction", () => {
	function setup(opts: {
		previousSummary?: string;
		checkpointInPrefix?: boolean;
		summaryMaxTokens?: unknown;
		reserveTokens?: number;
		modelMaxTokens?: number;
		response?: unknown;
		responses?: unknown[];
	} = {}) {
		const cwd = mkdtempSync(join(tmp, "summary-budget-"));
		mkdirSync(join(cwd, ".pi"));
		writeFileSync(join(cwd, ".pi", "settings.json"), JSON.stringify({
			betterCompaction: opts.summaryMaxTokens === undefined ? {} : { summaryMaxTokens: opts.summaryMaxTokens },
		}));
		const { entries } = makeEntries();
		let leafId = "entry-4";
		let keptEntryId = "entry-2";
		if (opts.previousSummary && opts.checkpointInPrefix !== false) {
			entries.push({
				type: "compaction", id: "previous-compaction", parentId: leafId,
				timestamp: new Date(0).toISOString(), summary: opts.previousSummary,
				firstKeptEntryId: "entry-2", tokensBefore: 100_000,
			});
			entries.push({
				type: "message", id: "recent", parentId: "previous-compaction", timestamp: new Date(1).toISOString(),
				message: { role: "user", content: "Keep this most recent message verbatim", timestamp: 1 },
			});
			leafId = "recent";
			keptEntryId = "entry-4";
		}
		const api = freshExtension();
		const base = makeCtx({
			sessionId: "summary-budget", cwd, entries, leafId, notify: () => {},
			messages: opts.responses ?? (opts.response ? [opts.response] : []),
			streamFactory: (message) => ({ async *[Symbol.asyncIterator]() { yield { type: "done", message }; } }),
		});
		const ctx = { ...base, model: { ...base.model, maxTokens: opts.modelMaxTokens ?? 128_000 } };
		const event = {
			...beforeCompactEvent(keptEntryId),
			preparation: {
				...makePreparation(keptEntryId), previousSummary: opts.previousSummary,
				settings: { reserveTokens: opts.reserveTokens ?? 20_000 },
			},
		};
		return { api, ctx, event };
	}

	async function compact(opts?: Parameters<typeof setup>[0]) {
		const run = setup(opts);
		const result = await fire(run.api, "session_before_compact", run.event, run.ctx);
		const request = run.ctx.calls[0]![1] as TranscriptContext;
		const options = run.ctx.calls[0]![2] as SimpleStreamOptions;
		const prompt = JSON.stringify(request.messages.at(-1));
		return { ...run, result, request, options, prompt };
	}

	test("reuses an oversized checkpoint once in the untouched prefix, without copying it into the final prompt", async () => {
		const previousSummary = "## Goal\nUNIQUE_OVERSIZED_CHECKPOINT\n" + "Historical experiment and artifact detail.\n".repeat(2500);
		const { ctx, event, result, request, prompt } = await compact({ previousSummary });
		const projection = buildSessionProjection(ctx.sessionManager.getEntries() as SessionEntry[], ctx.sessionManager.getLeafId());
		const cut = projection.entries.findIndex((entry) => entry.sourceEntry.id === event.preparation.firstKeptEntryId);
		const originalPrefix = convertToLlm(projection.entries.slice(0, cut).flatMap((entry) => entry.messages));
		expect(request.messages.slice(0, -1)).toEqual(originalPrefix);
		expect(JSON.stringify(request.messages).split("UNIQUE_OVERSIZED_CHECKPOINT")).toHaveLength(2);
		expect(prompt).not.toContain("UNIQUE_OVERSIZED_CHECKPOINT");
		expect(prompt).not.toContain("<previous-summary>");
		expect(prompt.length).toBeLessThan(7000);
		expect(prompt).toContain("An earlier compaction checkpoint is already provided above");
		expect(prompt).toContain("An oversized previous checkpoint must be condensed");
		expect(prompt).not.toContain("PRESERVE all existing information");
		expect(result).toHaveProperty("compaction.details.previousSummarySource", "prefix");
	});

	test("keeps a previous-summary fallback only when it is absent from the projected prefix", async () => {
		const previousSummary = "## Goal\nFALLBACK_CHECKPOINT_NEEDS_TO_SURVIVE";
		const { result, request, prompt } = await compact({ previousSummary, checkpointInPrefix: false });
		expect(JSON.stringify(request.messages).split("FALLBACK_CHECKPOINT_NEEDS_TO_SURVIVE")).toHaveLength(2);
		expect(prompt).toContain("<previous-summary>");
		expect(prompt).toContain("FALLBACK_CHECKPOINT_NEEDS_TO_SURVIVE");
		expect(result).toHaveProperty("compaction.details.previousSummarySource", "appended");
	});

	test("first compaction requests a concise complete checkpoint, not a historical artifact catalog", async () => {
		const { prompt, result } = await compact();
		expect(prompt).toContain("aim for at most 4096 tokens");
		expect(prompt).toContain("Keep the Done section to a few consolidated outcomes");
		expect(prompt).toContain("Do not generate <read-files> or <modified-files> lists");
		expect(prompt).not.toContain("An earlier compaction checkpoint");
		expect(result).toHaveProperty("compaction.details.previousSummarySource", "none");
	});

	test.each([0, 20_000, 400_000])("large or zero reserveTokens=%i never changes the summary budget", async (reserveTokens) => {
		const { api, ctx, options, result } = await compact({ reserveTokens });
		expect(options.maxTokens).toBe(8192);
		expect(result).toHaveProperty("compaction.details.summaryBudget", { targetTokens: 4096, maxTokens: 8192 });
		const notices: string[] = [];
		ctx.ui.notify = (message: string) => { notices.push(message); };
		await api.commands.get("bc")!.handler("", ctx);
		expect(notices.at(-1)).toContain("summary budget: 8192 tokens");
	});

	test.each([
		{ summaryMaxTokens: 2048, modelMaxTokens: 128_000, maxTokens: 2048, target: 1024 },
		{ summaryMaxTokens: 16_384, modelMaxTokens: 128_000, maxTokens: 16_384, target: 8192 },
		{ summaryMaxTokens: 8192, modelMaxTokens: 2048, maxTokens: 2048, target: 1024 },
		{ summaryMaxTokens: 8192, modelMaxTokens: 0, maxTokens: 8192, target: 4096 },
	])("honors configured and model budgets: %j", async ({ summaryMaxTokens, modelMaxTokens, maxTokens, target }) => {
		const { options, prompt } = await compact({ summaryMaxTokens, modelMaxTokens });
		expect(options.maxTokens).toBe(maxTokens);
		expect(prompt).toContain(`aim for at most ${target} tokens`);
	});

	test.each([null, 0, -1, 512, 1024.5, "8192", Number.MAX_SAFE_INTEGER + 1])("rejects invalid summaryMaxTokens=%j before requesting a summary", async (summaryMaxTokens) => {
		const { api, ctx, event } = setup({ summaryMaxTokens });
		await expect(fire(api, "session_before_compact", event, ctx)).rejects.toThrow("summaryMaxTokens must be a safe integer of at least 1024");
		expect(ctx.calls).toHaveLength(0);
		expect(api.entries).toEqual([]);
	});

	test("keeps complete file-operation bookkeeping outside the model's text budget", async () => {
		const { api, ctx, event } = setup();
		event.preparation.fileOps.read.add("README.md");
		event.preparation.fileOps.written.add("src/important.ts");
		const result = await fire(api, "session_before_compact", event, ctx);
		expect(result).toHaveProperty("compaction.summary", "## Goal\nKeep the test green.\n\n<read-files>\nREADME.md\n</read-files>\n\n<modified-files>\nsrc/important.ts\n</modified-files>");
		expect(result).toHaveProperty("compaction.details.modifiedFiles", ["src/important.ts"]);
	});

	test("uses a projected checkpoint even when preparation omits previousSummary", async () => {
		const { api, ctx, event } = setup({ previousSummary: "PROJECTED_CHECKPOINT" });
		event.preparation.previousSummary = undefined;
		const result = await fire(api, "session_before_compact", event, ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(JSON.stringify(request.messages.at(-1))).toContain("An earlier compaction checkpoint");
		expect(JSON.stringify(request.messages).split("PROJECTED_CHECKPOINT")).toHaveLength(2);
		expect(result).toHaveProperty("compaction.details.previousSummarySource", "prefix");
	});

	test("never resurrects a checkpoint explicitly omitted by a context edit", async () => {
		const { api, ctx, event } = setup({ previousSummary: "OMITTED_CHECKPOINT" });
		ctx.sessionManager.getEntries().push({
			type: "context_edit", id: "omit-checkpoint", parentId: "recent", timestamp: new Date(2).toISOString(),
			targetId: "previous-compaction", replacement: null,
		});
		ctx.sessionManager.getLeafId = () => "omit-checkpoint";
		const result = await fire(api, "session_before_compact", event, ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(JSON.stringify(request.messages)).not.toContain("OMITTED_CHECKPOINT");
		expect(result).toHaveProperty("compaction.details.previousSummarySource", "omitted");
	});

	test("neither an inactive branch nor summary-looking user text counts as a projected checkpoint", async () => {
		const { api, ctx, event } = setup();
		const entries = ctx.sessionManager.getEntries();
		(entries[0] as { message: { content: string } }).message.content = "<summary>Ordinary user text</summary>";
		entries.push({
			type: "compaction", id: "other-branch", parentId: "entry-1", timestamp: new Date(2).toISOString(),
			summary: "INACTIVE_BRANCH_CHECKPOINT", firstKeptEntryId: "entry-0", tokensBefore: 10_000,
		});
		const result = await fire(api, "session_before_compact", event, ctx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		expect(JSON.stringify(request.messages)).not.toContain("INACTIVE_BRANCH_CHECKPOINT");
		expect(JSON.stringify(request.messages.at(-1))).not.toContain("An earlier compaction checkpoint");
		expect(result).toHaveProperty("compaction.details.previousSummarySource", "none");
	});

	test("two extension compactions preserve cumulative tracking without copying file-list XML", async () => {
		const first = setup();
		first.event.preparation.fileOps.read.add("unchanged.md");
		first.event.preparation.fileOps.read.add("converted.ts");
		first.event.preparation.fileOps.written.add("historical-mod.ts");
		const firstResult = await fire(first.api, "session_before_compact", first.event, first.ctx) as {
			compaction: { summary: string; details: unknown };
		};
		const second = setup({ previousSummary: firstResult.compaction.summary });
		const checkpoint = second.ctx.sessionManager.getEntries().find((entry) =>
			(entry as { id: string }).id === "previous-compaction") as { details?: unknown; fromHook?: boolean };
		checkpoint.details = firstResult.compaction.details;
		checkpoint.fromHook = true;
		second.event.preparation.fileOps.read.add("historical-mod.ts");
		second.event.preparation.fileOps.written.add("converted.ts");
		second.event.preparation.fileOps.written.add("new.ts");
		const result = await fire(second.api, "session_before_compact", second.event, second.ctx);
		expect(result).toHaveProperty("compaction.details.readFiles", ["unchanged.md"]);
		expect(result).toHaveProperty("compaction.details.modifiedFiles", ["converted.ts", "historical-mod.ts", "new.ts"]);
		expect(result).toHaveProperty("compaction.summary", "## Goal\nKeep the test green.\n\n<read-files>\nunchanged.md\n</read-files>\n\n<modified-files>\nconverted.ts\nhistorical-mod.ts\nnew.ts\n</modified-files>");
		expect([...second.event.preparation.fileOps.read]).toEqual(["historical-mod.ts"]);
		expect([...second.event.preparation.fileOps.written]).toEqual(["converted.ts", "new.ts"]);
	});

	test("a summary target never shrinks a legacy chat-template thinking budget relative to the normal request", async () => {
		const { api, ctx, event } = setup({ reserveTokens: 400_000 });
		const model: Model<"openai-completions"> = {
			api: "openai-completions", provider: "test", id: "budget-model", name: "Budget model",
			baseUrl: "https://example.invalid/v1", reasoning: true, input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 1_000_000, maxTokens: 128_000,
			compat: { thinkingFormat: "chat-template", chatTemplateKwargs: { thinking_budget: { $var: "thinking.budget" } } },
		};
		const thinkingCtx = { ...ctx, model, thinkingLevel: "high" };
		await fire(api, "session_before_compact", event, thinkingCtx);
		const request = ctx.calls[0]![1] as TranscriptContext;
		const options = ctx.calls[0]![2] as SimpleStreamOptions;
		expect(options.maxTokens).toBe(128_000);
		expect(JSON.stringify(request.messages.at(-1))).toContain("aim for at most 4096 tokens");
		const { streamSimple } = await import("@earendil-works/pi-ai/api/openai-completions");
		async function capture(callOptions: SimpleStreamOptions, context: TranscriptContext) {
			let wire: unknown;
			const result = await streamSimple(model, context, {
				...callOptions, apiKey: "synthetic-key",
				onPayload: (body) => { wire = body; throw new Error("captured-before-network"); },
			}).result();
			expect(result.errorMessage).toContain("captured-before-network");
			return wire as { chat_template_kwargs: { thinking_budget: number } };
		}
		const normalContext = normalizeContext({ messages: convertToLlm(buildSessionProjection(
			ctx.sessionManager.getEntries() as SessionEntry[], ctx.sessionManager.getLeafId(),
		).messages) });
		const normal = await capture({ reasoning: "high", cacheRetention: "short" }, normalContext);
		const summary = await capture(options, request);
		expect(normal.chat_template_kwargs.thinking_budget).toBe(16_384);
		expect(summary.chat_template_kwargs).toEqual(normal.chat_template_kwargs);
	});

	test("Anthropic keeps the retained tail so its cache entry matches, scoping the summary to the cut", async () => {
		const { api, ctx, event } = setup();
		const model = { ...ctx.model, api: "anthropic-messages", provider: "anthropic", reasoning: false };
		await fire(api, "session_before_compact", event, { ...ctx, model });
		const messages = (ctx.calls[0]![1] as TranscriptContext).messages;
		const text = (i: number) => JSON.stringify(messages[i]);
		expect(messages).toHaveLength(6);
		expect(text(0)).toContain("Read the codebase");
		expect(text(2)).toContain("Now add a config file option.");
		expect(text(4)).toContain("Great, commit it.");
		const prompt = text(5);
		expect(prompt).toContain("Summarize ONLY the messages BEFORE the boundary");
		expect(prompt).toContain('the user message starting with: \\"Now add a config file option.\\"');
	});

	test("other providers send only the prefix before the cut, with no scope note", async () => {
		const { api, ctx, event } = setup();
		await fire(api, "session_before_compact", event, ctx);
		const messages = (ctx.calls[0]![1] as TranscriptContext).messages;
		expect(messages).toHaveLength(3);
		expect(JSON.stringify(messages)).not.toContain("Now add a config file option.");
		expect(JSON.stringify(messages.at(-1))).not.toContain("SCOPE:");
	});

	test.each(["text", "tool", "mixed"])("never persists a token-capped partial checkpoint (%s)", async (kind) => {
		const response = makeSummaryMessage({ stopReason: "length", text: kind === "tool" ? "" : "## Goal\nIncomplete checkpoint" });
		const { api, ctx, event } = setup({ response: kind !== "text" ? {
			...response, content: [...response.content, { type: "toolCall", id: "call_1", name: "read", arguments: {} }],
		} : response });
		await expect(fire(api, "session_before_compact", event, ctx)).rejects.toThrow("generation hit the token cap");
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toEqual([]);
		expect(ctx.calls).toHaveLength(1);
	});

	test("rejects a truncated recovery attempt after an initial tool-only response", async () => {
		const first = { ...makeSummaryMessage(), content: [{ type: "toolCall", id: "call_1", name: "read", arguments: {} }] };
		const second = makeSummaryMessage({ stopReason: "length", text: "Incomplete replacement" });
		const { api, ctx, event } = setup({ responses: [first, second] });
		await expect(fire(api, "session_before_compact", event, ctx)).rejects.toThrow("generation hit the token cap");
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toEqual([]);
		expect(ctx.calls).toHaveLength(2);
	});
});

describe("append-only Anthropic compaction effort", () => {
	const beta = "mid-conversation-output-config-2026-07-01";
	const marker = (effort = "max") => ({ role: "system", content: [], output_config: { effort } });
	const summary = { role: "user", content: [{ type: "text", text: "Summarize this conversation" }] };
	const payload = (messages: unknown[] = [summary, marker()]) => ({
		model: "claude-opus-5-5", messages, max_tokens: 8192, stream: true,
		betas: [beta, "thinking-binding-controls-2026-08-01"],
		thinking: { type: "adaptive", display: "summarized", block_binding: { prefix_mismatch_behavior: "drop_block" } },
		output_config: { effort: "high" },
		system: [{ type: "text", text: "Original instructions", cache_control: { type: "ephemeral" } }],
	});

	async function prepare(
		overrides: Partial<Model<Api>> = {},
		thinking: "off" | "inherit" = "off",
		sessionThinking = "max",
	) {
		const cwd = mkdtempSync(join(tmp, "anthropic-effort-"));
		mkdirSync(join(cwd, ".pi"));
		writeFileSync(join(cwd, ".pi", "settings.json"), JSON.stringify({ betterCompaction: { thinking } }));
		const api = freshExtension();
		const notices: string[] = [];
		const model: Model<Api> = {
			api: "anthropic-messages", provider: "anthropic", id: "claude-opus-5-5", name: "Test Opus",
			baseUrl: "https://api.anthropic.com", reasoning: true, input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 1_000_000, maxTokens: 128_000,
			thinkingLevelMap: { off: null, minimal: null, xhigh: "xhigh", max: "max" },
			compat: { supportsMidConvoEffort: true, forceAdaptiveThinking: true },
			...overrides,
		};
		const { entries, leafId } = makeEntries();
		for (const entry of entries as SessionEntry[]) {
			if (entry.type !== "message" || entry.message.role !== "assistant") continue;
			const message = entry.message as AssistantMessage;
			Object.assign(message, { api: model.api, provider: model.provider, model: model.id, providerThinkingLevel: "xhigh" });
			message.content.unshift({ type: "thinking", thinking: "Earlier reasoning", thinkingSignature: "synthetic-signature" });
		}
		entries.unshift({
			type: "message", id: "system-0", parentId: null, timestamp: new Date(0).toISOString(),
			message: { role: "system", content: "Original instructions", timestamp: 0 },
		});
		(entries[1] as { parentId: string | null }).parentId = "system-0";
		const ctx = {
			...makeCtx({
				sessionId: "anthropic-effort", cwd, entries, leafId, messages: [],
				notify: (message) => notices.push(message),
				streamFactory: (message) => ({ async *[Symbol.asyncIterator]() { yield { type: "done", message }; } }),
			}),
			model, thinkingLevel: sessionThinking,
		};
		await fire(api, "before_agent_start", { systemPromptOptions: {} }, ctx);
		await api.commands.get("bc")!.handler(`thinking ${thinking}`, ctx);
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		return { api, ctx, notices, result, context: ctx.calls[0]![1] as TranscriptContext, options: ctx.calls[0]![2] as SimpleStreamOptions };
	}

	test("changes only the terminal effort marker, preserving signed history, caches, and top-level thinking", async () => {
		const { api, ctx, options, notices, result } = await prepare();
		expect(options.reasoning).toBe("max");
		expect(notices[0]).toContain("per-message effort: low");
		await api.commands.get("bc")!.handler("", ctx);
		expect(notices.at(-1)).toContain("compaction thinking: off (per-message effort: low)");
		await fire(api, "session_start", {}, ctx);
		expect(notices.at(-1)).toContain("per-message effort: low");
		expect(result).toHaveProperty("compaction.details.reasoningUpdate", { kind: "anthropic", effort: "low" });

		const prefix = [
			{ role: "user", content: [{ type: "text", text: "Original request", cache_control: { type: "ephemeral" } }] },
			marker("xhigh"),
			{ role: "assistant", content: [
				{ type: "thinking", thinking: "Earlier reasoning", signature: "unchanged-signature" },
				{ type: "tool_use", id: "call_1", name: "read", input: { path: "README.md" } },
			] },
			{ role: "user", content: [{ type: "tool_result", tool_use_id: "call_1", content: "Original result" }] },
			marker("medium"),
			{ role: "assistant", content: [{ type: "text", text: "Earlier answer" }] },
		];
		const body = payload([...prefix, summary, marker()]);
		const original = structuredClone(body);
		function freeze(value: unknown) {
			if (value && typeof value === "object") {
				Object.freeze(value);
				for (const child of Object.values(value)) freeze(child);
			}
		}
		freeze(body);
		const updated = await options.onPayload!(body, ctx.model) as typeof body;
		expect(body).toEqual(original);
		expect(updated.messages).toEqual([...prefix, summary, marker("low")]);
		expect({ ...updated, messages: body.messages }).toEqual(body);
		for (let i = 0; i < prefix.length + 1; i++) expect(updated.messages[i]).toBe(body.messages[i]);
		expect(await options.onPayload!(updated, ctx.model)).toBe(updated);
	});

	test("gives adaptive Anthropic thinking headroom beyond the summary budget without raising the text target", async () => {
		const { options, result } = await prepare();
		expect(options.maxTokens).toBe(8192 + 16_384);
		expect(result).toHaveProperty("compaction.details.summaryBudget", { targetTokens: 4096, maxTokens: 8192 + 16_384 });
		const capped = await prepare({ maxTokens: 10_000 });
		expect(capped.options.maxTokens).toBe(10_000);
	});

	test("inherit keeps the native effort untouched", async () => {
		const { options } = await prepare({}, "inherit");
		expect(options.reasoning).toBe("max");
		expect(options.onPayload).toBeUndefined();
	});

	test.each([
		{ compat: undefined },
		{ compat: { supportsMidConvoEffort: false, forceAdaptiveThinking: true } },
		{ compat: { supportsMidConvoSystemMessages: true, forceAdaptiveThinking: true } },
		{ provider: "custom-proxy" },
		{ api: "bedrock-converse-stream", provider: "amazon-bedrock" },
		{ reasoning: false },
	] satisfies Partial<Model<Api>>[])("unverified model/transport stays prompt-only: %j", async (overrides) => {
		const { options, notices } = await prepare(overrides);
		expect(options.onPayload).toBeUndefined();
		expect(notices[0]).toContain("prompt only; API thinking unchanged");
	});

	test("an already-off session that supports disabled thinking does not enable adaptive thinking or effort markers", async () => {
		const { options, notices } = await prepare({ thinkingLevelMap: { max: "max" } }, "off", "off");
		expect(options.reasoning).toBeUndefined();
		expect(options.onPayload).toBeUndefined();
		expect(notices[0]).toContain("API thinking unchanged");
	});

	test("off-unsupported models may still lower effort without changing their implicit adaptive mode", async () => {
		const { options, ctx } = await prepare({}, "off", "off");
		expect(options.reasoning).toBeUndefined();
		const updated = await options.onPayload!(payload(), ctx.model) as ReturnType<typeof payload>;
		expect(updated.messages.at(-1)).toEqual(marker("low"));
		expect(updated.thinking).toEqual(payload().thinking);
	});

	test.each(["disabled", "between_tools", "enabled"])("does not rewrite a non-adaptive %s payload", async (type) => {
		const { options, ctx } = await prepare();
		const body = { ...payload(), thinking: { type } };
		const original = structuredClone(body);
		expect(() => options.onPayload!(body, ctx.model)).toThrow("expected adaptive Anthropic thinking");
		expect(body).toEqual(original);
	});

	test("rejects missing beta, malformed/ambiguous suffixes, and tool-result-only user turns", async () => {
		const { options, ctx } = await prepare();
		const body = payload();
		const badBodies = [
			{}, { ...body, messages: [] }, { ...body, messages: "invalid" },
			{ ...body, betas: undefined }, { ...body, betas: ["unrelated-beta"] },
			...[ [summary], [marker()], [summary, marker("none")],
				[summary, { ...marker(), content: [{ type: "text", text: "Not an effort-only marker" }] }],
				[summary, { ...marker(), unexpected: true }],
				[summary, { ...marker(), output_config: { effort: "max", other: true } }],
				[summary, { role: "assistant", content: [] }],
				[summary, marker(), marker()],
				[{ role: "user", content: " " }, marker()],
				[{ role: "user", content: [{ type: "tool_result", tool_use_id: "call_1", content: "Result" }] }, marker()],
			].map((messages) => ({ ...body, messages })),
		];
		for (const badBody of badBodies) {
			const original = structuredClone(badBody);
			expect(() => options.onPayload!(badBody, ctx.model)).toThrow("Better Compaction: expected");
			expect(badBody).toEqual(original);
		}
	});

	test("real Anthropic adapter changes only the summary marker, not the cached main-request prefix", async () => {
		const { ctx, context, options } = await prepare();
		const { streamSimple } = await import("@earendil-works/pi-ai/api/anthropic-messages");
		const model = ctx.model as Model<"anthropic-messages">;
		type Wire = ReturnType<typeof payload>;
		async function capture(request: TranscriptContext, inject: boolean, normalRequest = false) {
			let wire: unknown;
			const result = await streamSimple(model, request, {
				...options, maxTokens: normalRequest ? undefined : options.maxTokens, apiKey: "synthetic-test-key",
				onPayload: async (body, m) => {
					wire = inject ? await options.onPayload!(body, m) : body;
					throw new Error("captured-before-network");
				},
			}).result();
			expect(result.errorMessage).toContain("captured-before-network");
			expect(wire).toBeDefined();
			return wire as Wire;
		}
		const mainContext = normalizeContext({ messages: convertToLlm(buildSessionProjection(
			ctx.sessionManager.getEntries() as SessionEntry[], ctx.sessionManager.getLeafId(),
		).messages) });
		const main = await capture(mainContext, false, true);
		const baseline = await capture(context, false);
		const updated = await capture(context, true);
		expect(baseline.messages.at(-1)).toEqual(marker("max"));
		expect(baseline.output_config).toEqual({ effort: "high" });
		expect(baseline.betas).toContain(beta);
		expect(baseline.thinking.type).toBe("adaptive");
		expect(updated.messages).toEqual([...baseline.messages.slice(0, -1), marker("low")]);
		expect({ ...updated, messages: baseline.messages }).toEqual(baseline);
		// Anthropic reads only entries written at earlier breakpoints, so the
		// summary request keeps the retained tail: the agent's whole last request
		// is an exact prefix (modulo where its cache breakpoint sat), and the
		// cut-only prefix is NOT what is sent.
		const plain = (messages: unknown[]) => (JSON.parse(JSON.stringify(messages), (key, value) =>
			key === "cache_control" ? undefined : value) as { content: unknown }[]).map((m) => ({
			...m, content: typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content,
		}));
		// The agent's own trailing effort marker is replaced by the summary turn's.
		expect(main.messages.at(-1)).toEqual(marker("max"));
		expect(updated.messages).toHaveLength(main.messages.length + 1);
		expect(plain(updated.messages.slice(0, -2))).toEqual(plain(main.messages.slice(0, -1)));
		expect(JSON.stringify(updated.messages.at(-2))).toContain("Summarize ONLY the messages BEFORE the boundary");
		expect(JSON.stringify(updated.messages.at(-2))).toContain("Now add a config file option.");
		expect(JSON.stringify(updated.messages.slice(0, -2))).toContain("synthetic-signature");
		expect(JSON.stringify(updated.messages.slice(0, -2))).toContain('"effort":"xhigh"');
		expect({ ...updated, messages: main.messages, max_tokens: main.max_tokens }).toEqual(main);
	});
});

describe("append-only OpenAI compaction reasoning", () => {
	async function prepare(
		overrides: Partial<Model<Api>> = {},
		thinking: "off" | "inherit" = "off",
	) {
		const cwd = mkdtempSync(join(tmp, "reasoning-"));
		mkdirSync(join(cwd, ".pi"));
		const api = freshExtension();
		const notices: string[] = [];
		const model: Model<Api> = {
			api: "openai-codex-responses",
			provider: "openai-codex",
			id: "gpt-6.1-terra",
			name: "Test model",
			baseUrl: "https://chatgpt.com/backend-api",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1_050_000,
			maxTokens: 8192,
			thinkingLevelMap: { off: "none", max: "max" },
			...overrides,
		};
		const ctx = {
			...makeCtx({
				sessionId: "reasoning-test", cwd, notify: (message) => notices.push(message), messages: [],
				streamFactory: (message) => ({ async *[Symbol.asyncIterator]() { yield { type: "done", message }; } }),
			}),
			model,
			thinkingLevel: "max",
		};
		await api.commands.get("bc")!.handler(`thinking ${thinking}`, ctx);
		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		return {
			api, ctx, notices,
			context: ctx.calls[0]![1] as TranscriptContext,
			options: ctx.calls[0]![2] as SimpleStreamOptions,
		};
	}

	test("thinking off keeps max top-level reasoning and inserts none immediately before the summary", async () => {
		const { api, ctx, notices, options } = await prepare();
		expect(options.reasoning).toBe("max");
		expect(notices[0]).toContain("configuration_update: none");
		await api.commands.get("bc")!.handler("", ctx);
		expect(notices.at(-1)).toContain("compaction thinking: off (configuration_update: none)");
		const prefix = [
			{ type: "message", role: "user", content: [{ type: "input_text", text: "Original request" }] },
			{ type: "reasoning", id: "rs_1", encrypted_content: "unchanged" },
			{ type: "function_call", call_id: "call_1", name: "read", arguments: "{}" },
			{ type: "function_call_output", call_id: "call_1", output: "Original result" },
		];
		const prompt = { role: "user", content: [{ type: "input_text", text: "Summarize" }] };
		const body = {
			input: [...prefix, prompt], reasoning: { effort: "max", summary: "auto" },
			instructions: "Original instructions", prompt_cache_key: "original-key", store: false,
		};
		const original = structuredClone(body);
		const updated = await options.onPayload!(body, ctx.model) as Omit<typeof body, "input"> & { input: unknown[] };
		expect(body).toEqual(original); // No mutation of a cached/reused payload.
		expect(updated.input).toEqual([
			...prefix, { type: "configuration_update", reasoning: { effort: "none" } }, prompt,
		]);
		expect({ ...updated, input: body.input }).toEqual(body); // No other wire settings change.
		for (let i = 0; i < prefix.length; i++) expect(updated.input[i]).toBe(prefix[i]!);
		expect(updated.input.at(-1)).toBe(prompt);
		const replay = await options.onPayload!(updated, ctx.model);
		expect(replay).toBe(updated); // No adjacent/duplicate updates on a repeated hook.
	});

	test("models without none/minimal receive their lowest supported effort", async () => {
		const { ctx, notices, options } = await prepare({
			id: "gpt-6.1-sol", thinkingLevelMap: { off: null, minimal: null, max: "max" },
		});
		expect(options.reasoning).toBe("max");
		expect(notices[0]).toContain("configuration_update: low");
		const body = await options.onPayload!({ input: [{ role: "user", content: "Summary" }] }, ctx.model) as { input: unknown[] };
		expect(body.input[0]).toEqual({ type: "configuration_update", reasoning: { effort: "low" } });
	});

	test("uses a model's wire effort mapping rather than its Pi level name", async () => {
		const { ctx, options } = await prepare({ thinkingLevelMap: { off: null, minimal: "low", max: "max" } });
		const body = await options.onPayload!({ input: [{ role: "user", content: "Summary" }] }, ctx.model) as { input: unknown[] };
		expect(body.input[0]).toEqual({ type: "configuration_update", reasoning: { effort: "low" } });
	});

	test("inherit does not inject an effort override", async () => {
		const { options } = await prepare({}, "inherit");
		expect(options.reasoning).toBe("max");
		expect(options.onPayload).toBeUndefined();
	});

	test.each([
		{ id: "gpt-5.5" },
		{ id: "gpt-60" },
		{ provider: "custom-proxy" },
		{ api: "openai-completions" },
		{ api: "anthropic-messages", provider: "anthropic", id: "claude-opus" },
		{ reasoning: false },
	] satisfies Partial<Model<Api>>[])("unsupported model uses explicit prompt-only fallback: %j", async (overrides) => {
		const { context, notices, options } = await prepare(overrides);
		expect(options.onPayload).toBeUndefined();
		expect(notices[0]).toContain("prompt only; API thinking unchanged");
		expect(JSON.stringify(context.messages.at(-1))).toContain("Thinking is disabled for this task");
	});

	test("rejects malformed payloads rather than silently claiming to disable thinking", async () => {
		const { ctx, options } = await prepare();
		for (const body of [{}, { input: [] }, { input: [{ role: "assistant" }] }]) {
			expect(() => options.onPayload!(body, ctx.model)).toThrow("Better Compaction: expected");
		}
		expect(() => options.onPayload!({ input: [
			{ type: "configuration_update", reasoning: { effort: "high" } }, { role: "user" },
		] }, ctx.model)).toThrow("cannot append adjacent");
	});

	test.each(["openai-responses", "openai-codex-responses"] as const)(
		"real %s adapter emits the update while preserving the rest of its wire request", async (api) => {
			const { ctx, context, options } = await prepare({
				api, provider: api === "openai-responses" ? "openai" : "openai-codex",
				baseUrl: api === "openai-responses" ? "https://api.openai.com/v1" : "https://chatgpt.com/backend-api",
			});
			const codex = await import("@earendil-works/pi-ai/api/openai-codex-responses");
			const responses = await import("@earendil-works/pi-ai/api/openai-responses");
			// Synthetic credentials satisfy local validation only. The capture hook
			// aborts before any network request, including WebSocket connection.
			const jwt = `test.${Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "test" } })).toString("base64url")}.test`;
			async function capture(inject: boolean) {
				let wire: unknown;
				const streamOptions: SimpleStreamOptions = {
					...options, apiKey: api === "openai-responses" ? "test" : jwt,
					onPayload: async (payload, model) => {
						wire = inject ? await options.onPayload!(payload, model) : payload;
						throw new Error("captured-before-network");
					},
				};
				const stream = api === "openai-responses"
					? responses.streamSimple({ ...ctx.model, api: "openai-responses" }, context, streamOptions)
					: codex.streamSimple({ ...ctx.model, api: "openai-codex-responses" }, context, streamOptions);
				const result = await stream.result();
				expect(result.errorMessage).toContain("captured-before-network");
				expect(wire).toBeDefined();
				return wire as { input: unknown[]; reasoning: { effort: string } };
			}
			const baseline = await capture(false);
			const updated = await capture(true);
			expect(baseline.reasoning.effort).toBe("max");
			expect(updated.reasoning).toEqual(baseline.reasoning);
			expect(updated.input.slice(0, -2)).toEqual(baseline.input.slice(0, -1));
			expect(updated.input.at(-2)).toEqual({ type: "configuration_update", reasoning: { effort: "none" } });
			expect(updated.input.at(-1)).toEqual(baseline.input.at(-1));
			expect({ ...updated, input: baseline.input }).toEqual(baseline);
		},
	);
});

describe("post-compaction report lifecycle", () => {
	test("manual compaction still works when Pi auto-compaction is disabled", async () => {
		const cwd = join(tmp, "manual-auto-disabled");
		mkdirSync(join(cwd, ".pi"), { recursive: true });
		writeFileSync(join(cwd, ".pi", "settings.json"), JSON.stringify({ compaction: { enabled: false } }));
		const api = freshExtension();
		const ctx = makeCtx({ sessionId: "manual-auto-disabled", cwd, notify: () => {}, messages: [] });
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toHaveProperty("compaction");
		expect(ctx.calls).toHaveLength(1);
	});

	test("summary requests stay short even when the environment requests long caching", async () => {
		const retention = process.env.PI_CACHE_RETENTION;
		try {
			process.env.PI_CACHE_RETENTION = "long";
			const api = freshExtension();
			const ctx = makeCtx({ sessionId: "short-summary", cwd: projectDir, notify: () => {}, messages: [] });
			await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
			expect(ctx.calls[0]![2]).toMatchObject({ cacheRetention: "short" });
		} finally {
			if (retention === undefined) delete process.env.PI_CACHE_RETENTION;
			else process.env.PI_CACHE_RETENTION = retention;
		}
	});

	test("is NOT emitted during session_before_compact, only on session_compact", async () => {
		const api = freshExtension();
		const ctx = makeCtx({ sessionId: "s1", cwd: projectDir, notify: () => {}, messages: [] });

		// Summarization runs and the compaction result is returned…
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toBeDefined();
		const compaction = (result as { compaction: Record<string, unknown> }).compaction;
		expect(compaction.details).toMatchObject({ strategy: "better-compaction" });
		expect(compaction.firstKeptEntryId).toBe("entry-2");
		expect(compaction.tokensBefore).toBe(123_456);
		expect(String(compaction.summary)).toContain("## Goal");
		expect(ctx.calls[0]![2]).toMatchObject({ cacheRetention: "short" });
		// …but nothing is reported yet: the compaction is not complete.
		expect(api.entries).toEqual([]);

		// …and the report entry appears exactly when pi says the compaction
		// succeeded, with the final attempt's numbers.
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toHaveLength(1);
		expect(api.entries[0]!.customType).toBe("better-compaction-report");
		expect(api.entries[0]!.data).toMatchObject({ cacheRead: 12_345 });

		// …and never again for the same compaction.
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toHaveLength(1);
	});

	test("reports the final attempt's timing after a retried summarization", async () => {
		const api = freshExtension();
		const messages: string[] = [];
		const cwd = join(tmp, "retry-project");
		mkdirSync(join(cwd, ".pi"), { recursive: true });
		// Fast retry so the test stays quick.
		writeFileSync(
			join(cwd, ".pi", "settings.json"),
			JSON.stringify({ retry: { enabled: true, maxRetries: 3, baseDelayMs: 10, maxAgentDelayMs: 100 } }),
		);
		// First attempt fails with a retryable error, second succeeds with
		// different (final) usage — the report must reflect the final attempt.
		const ctx = makeCtx({
			sessionId: "s2",
			cwd,
			notify: (m) => messages.push(m),
			messages: [
				makeSummaryMessage({ stopReason: "error", errorMessage: "overloaded, please retry" }),
				makeSummaryMessage({ cacheRead: 55, text: "## Goal\nFinal attempt summary." }),
			],
		});

		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect((result as { compaction: { summary: string } }).compaction.summary).toContain("Final attempt");
		expect(api.entries).toEqual([]);

		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toHaveLength(1);
		expect(api.entries[0]!.customType).toBe("better-compaction-report");
		// Must be the final attempt's usage (55), not attempt 1's 12_345.
		expect(api.entries[0]!.data).toMatchObject({ cacheRead: 55 });
	});

	test("does not report for pi's built-in compaction (no stashed report)", async () => {
		const api = freshExtension();
		const ctx = makeCtx({ sessionId: "s3", cwd: projectDir, notify: () => {}, messages: [] });

		// session_compact without a preceding session_before_compact result
		// (i.e. pi's built-in compaction ran) → no report.
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toEqual([]);
	});

	test("discards the stashed report when the compaction fails", async () => {
		const api = freshExtension();
		const ctx = makeCtx({ sessionId: "s4", cwd: projectDir, notify: () => {}, messages: [] });

		// Summarization succeeded (report stashed) but the compaction then fails.
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toBeDefined();
		await fire(api, "session_compact_failed", compactFailedEvent(), ctx);
		expect(api.entries).toEqual([]);

		// A later successful compaction in the same session must not emit the
		// stale report from the failed one (built-in path → nothing stashed).
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toEqual([]);
	});

	test("respects report=false (set at emit time)", async () => {
		const api = freshExtension();
		const cwd = join(tmp, "report-off");
		mkdirSync(join(cwd, ".pi"), { recursive: true });
		writeFileSync(join(cwd, ".pi", "settings.json"), JSON.stringify({ betterCompaction: { report: false } }));
		const ctx = makeCtx({ sessionId: "s5", cwd, notify: () => {}, messages: [] });

		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toBeDefined(); // summary is still produced…
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toEqual([]); // …but the report is suppressed
	});

	test("persists the report entry even when there is no UI", async () => {
		const api = freshExtension();
		const ctx = makeCtx({
			sessionId: "s6",
			cwd: projectDir,
			notify: () => {},
			messages: [],
			hasUI: false,
		});

		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toBeDefined();
		await fire(api, "session_compact", compactEvent(), ctx);
		// The report is a session entry, not a notification: no UI is needed,
		// and it renders whenever the session is viewed interactively.
		expect(api.entries).toHaveLength(1);
		expect(api.entries[0]!.customType).toBe("better-compaction-report");
	});

	test("keeps reports isolated per session", async () => {
		const api = freshExtension();
		const ctxA = makeCtx({ sessionId: "sA", cwd: projectDir, notify: () => {}, messages: [] });
		const ctxB = makeCtx({ sessionId: "sB", cwd: projectDir, notify: () => {}, messages: [] });

		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctxA);
		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctxB);

		// Session A's completion must not consume or duplicate session B's report.
		await fire(api, "session_compact", compactEvent(), ctxA);
		expect(api.entries).toHaveLength(1);

		await fire(api, "session_compact", compactEvent(), ctxB);
		expect(api.entries).toHaveLength(2);
	});

	test("closes the thinking span at text_start (OpenAI adapters emit thinking_end only at stream end)", async () => {
		const api = freshExtension();
		const ctx = makeCtx({
			sessionId: "s7",
			cwd: projectDir,
			notify: () => {},
			messages: [],
			streamFactory: makeOpenAiOrderStream,
		});

		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toHaveLength(1);
		const data = api.entries[0]!.data as { totalMs: number; thinkingMs: number };
		// The thinking span must end at text_start (~300ms), not at the late
		// thinking_end (~900ms): the 600ms tail is the answer being generated.
		// (With the old code thinkingMs ≈ 900 and "generating" ≈ 0.)
		expect(data.thinkingMs).toBeLessThan(600);
		expect(data.totalMs - data.thinkingMs).toBeGreaterThan(600);
	});
});

describe("edit-aware prefix rebuild (issue #3: context_edit projections)", () => {
	// The agent's LLM context is the edit-aware projection (buildSessionProjection),
	// which applies context_edit replacements/omissions — and that is the prefix
	// the backend cached. The rebuilt summarization prefix must be
	// byte-identical to it up to firstKeptEntryId, or prefix caching silently
	// stops matching at the first edited message and the summarizer reads
	// content the model never had.
	test("rebuilt prefix equals the projected agent context when context_edits precede the cut", async () => {
		const api = freshExtension();
		const { entries, leafId, keptEntryId } = makeEditedEntries();
		const ctx = makeCtx({ sessionId: "s8", cwd: projectDir, notify: () => {}, messages: [], entries, leafId });

		const result = await fire(api, "session_before_compact", beforeCompactEvent(keptEntryId), ctx);
		expect(result).toBeDefined();

		// The summarization request: modelRegistry.streamSimple(model, requestContext, options)
		const requestContext = ctx.calls[0]![1] as { messages: unknown[] };
		const sent = requestContext.messages.slice(0, -1); // minus the appended compaction prompt

		// Strong assertion: the sent prefix IS the projected agent context up
		// to the cut point (same conversion the projection itself uses).
		const projection = buildSessionProjection(entries as SessionEntry[], leafId);
		const cutIndex = projection.entries.findIndex((e) => e.sourceEntry.id === keptEntryId);
		expect(cutIndex).toBeGreaterThanOrEqual(0);
		const expectedPrefix = convertToLlm(projection.entries.slice(0, cutIndex).flatMap((e) => e.messages));
		expect(sent).toEqual(expectedPrefix);

		// Explicit, readable checks for the two edit kinds:
		const sentText = JSON.stringify(sent);
		// Omission: the omitted user message must not be sent at all…
		expect(sentText).not.toContain("Original omitted question.");
		// …it (and the two context_edit bookkeeping entries) contribute no
		// messages: only entry-0, entry-1 and the replaced entry-3 remain.
		expect(sent.length).toBe(3);
		// Replacement: the model sees the replacement, not the original content.
		expect(sentText).toContain("Shortened answer.");
		expect(sentText).not.toContain("Original long answer the model no longer sees.");
	});
});
