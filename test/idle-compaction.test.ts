/// <reference types="bun-types" />
import { describe, expect, test } from "bun:test";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
	registerIdleCompaction,
	type IdleCompactionClock,
	type IdleCompactionSettings,
} from "../extensions/better-compaction";

type Handler = (event: Record<string, unknown>, ctx: ExtensionContext) => unknown;

class Clock implements IdleCompactionClock {
	at = 1_000_000;
	private nextId = 0;
	private tasks = new Map<number, { at: number; callback: () => void }>();
	now = () => this.at;
	setTimer = (callback: () => void, delay: number) => {
		const id = ++this.nextId;
		this.tasks.set(id, { at: this.at + delay, callback });
		return () => {
			this.tasks.delete(id);
		};
	};
	get pending() {
		return this.tasks.size;
	}
	advance(ms: number) {
		const end = this.at + ms;
		for (;;) {
			const next = [...this.tasks].filter(([, task]) => task.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
			if (!next) break;
			this.at = next[1].at;
			this.tasks.delete(next[0]);
			next[1].callback();
		}
		this.at = end;
	}
	/** Simulate a sleeping machine: callbacks run only after wall time jumps. */
	wakeAfter(ms: number) {
		this.at += ms;
		for (const [id, task] of [...this.tasks]) {
			if (task.at <= this.at) {
				this.tasks.delete(id);
				task.callback();
			}
		}
	}
}

function payload(ttl: string | undefined = "1h") {
	return {
		messages: [
			{
				role: "user",
				content: [
					{ type: "text", text: "Conversation", cache_control: { type: "ephemeral", ...(ttl ? { ttl } : {}) } },
				],
			},
		],
	};
}

function harness(options: IdleCompactionSettings & { mode?: ExtensionContext["mode"]; id?: string } = {}) {
	const handlers = new Map<string, Handler[]>();
	const api = {
		on(name: string, handler: Handler) {
			const list = handlers.get(name) ?? [];
			list.push(handler);
			handlers.set(name, list);
			return () => {
				const index = list.indexOf(handler);
				if (index >= 0) list.splice(index, 1);
			};
		},
	};
	const clock = new Clock();
	const settings: IdleCompactionSettings = { idleCompaction: true, ...options };
	const compactions: Array<{ onError?: (error: Error) => void }> = [];
	const notices: string[] = [];
	const ctx = {
		mode: options.mode ?? "tui",
		model: {
			api: "anthropic-messages",
			provider: "anthropic",
			id: "claude-test",
			baseUrl: "https://api.anthropic.com",
			promptCache: {} as { short?: number; long?: number },
		},
		autoCompactionEnabled: true,
		busy: false,
		pending: false,
		tokens: 500_000 as number | null,
		contextWindow: 1_000_000,
		editor: "",
		thinkingLevel: "high",
		sessionManager: { getSessionId: () => options.id ?? "main" },
		getContextUsage: () => ({
			tokens: ctx.tokens,
			contextWindow: ctx.contextWindow,
			percent: ctx.tokens === null ? null : (ctx.tokens / ctx.contextWindow) * 100,
		}),
		isIdle: () => !ctx.busy,
		hasPendingMessages: () => ctx.pending,
		compact: (options: { onError?: (error: Error) => void }) => {
			compactions.push(options);
		},
		ui: {
			getEditorText: () => ctx.editor,
			notify: (message: string) => {
				notices.push(message);
			},
		},
	};
	const controller = registerIdleCompaction(
		api as unknown as ExtensionAPI, () => settings, clock, () => ctx.autoCompactionEnabled,
	);
	const fire = async (name: string, event: object = {}, context = ctx) => {
		let result: unknown;
		let current: Record<string, unknown> = { type: name, ...event };
		for (const handler of [...(handlers.get(name) ?? [])]) {
			const value = await handler(current, context as unknown as ExtensionContext);
			if (value !== undefined) {
				result = value;
				if (name === "before_provider_request") current = { ...current, payload: value };
			}
		}
		return result;
	};
	const start = () => fire("session_start");
	const request = async (body: unknown = payload(), status = 200) => {
		await fire("agent_start");
		await fire("context");
		await fire("before_provider_request", { payload: body });
		await fire("after_provider_response", { status });
	};
	const settle = () => fire("agent_settled");
	const refresh = () => controller.refresh(ctx as unknown as ExtensionContext);
	const status = () => controller.status(ctx as unknown as ExtensionContext);
	return { ctx, settings, clock, compactions, notices, start, request, settle, fire, refresh, status, on: api.on };
}

describe("idle compaction deadlines", () => {
	test("compacts just before an observed one-hour cache expires", async () => {
		const delay = 3_540_000;
		const h = harness();
		await h.start();
		await h.request(payload());
		await h.settle();
		expect(h.clock.pending).toBe(1);
		h.clock.advance(delay - 1);
		expect(h.compactions).toHaveLength(0);
		h.clock.advance(1);
		expect(h.compactions).toHaveLength(1);
		await h.settle();
		h.clock.advance(7_200_000);
		expect(h.compactions).toHaveLength(1);
	});

	test("anchors expiry to the request, not the end of generation", async () => {
		const h = harness();
		await h.start();
		await h.request();
		h.clock.advance(600_000);
		await h.settle();
		h.clock.advance(2_940_000);
		expect(h.compactions).toHaveLength(1);
	});

	test("uses a zero-delay timer if settlement falls inside the safety margin", async () => {
		const h = harness();
		await h.start();
		await h.request();
		h.clock.advance(3_570_000);
		await h.settle();
		expect(h.compactions).toHaveLength(0); // Notification-only event does not compact inline.
		h.clock.advance(0);
		expect(h.compactions).toHaveLength(1);
	});

	test("does not compact an expired cache after sleep or a slow run", async () => {
		for (const beforeSettle of [true, false]) {
			const h = harness();
			await h.start();
			await h.request();
			if (beforeSettle) h.clock.advance(3_600_000);
			await h.settle();
			h.clock.wakeAfter(3_600_000);
			expect(h.compactions).toHaveLength(0);
			expect(h.clock.pending).toBe(0);
		}
	});

	test("supports automatic top-level caching and conservative mixed conversation TTLs", async () => {
		const h = harness();
		await h.start();
		const body = { ...payload(), cache_control: { type: "ephemeral", ttl: "5m" } };
		await h.request(body);
		await h.settle();
		expect(h.clock.pending).toBe(0); // The shortest conversation breakpoint wins.
		expect(h.status()).toContain("short-term cache");
		h.clock.advance(240_000);
		expect(h.compactions).toHaveLength(0);
		await h.request({ cache_control: { type: "ephemeral", ttl: "1h" }, messages: [{ role: "user", content: "test" }] });
		await h.settle();
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(1);
	});

	test("observes replacement payloads from extensions loaded after BetterCompaction", async () => {
		const h = harness();
		const removeTransform = h.on("before_provider_request", () => payload("5m"));
		await h.start();
		await h.request(payload("1h"));
		await h.settle();
		expect(h.clock.pending).toBe(0);
		expect(h.status()).toContain("short-term cache");
		h.clock.advance(240_000);
		expect(h.compactions).toHaveLength(0);
		removeTransform();
		// Replacing observers must not leave an older observer before the new transform.
		h.on("before_provider_request", () => ({ messages: [] }));
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(0);
	});

	test("reads in-place TTL changes made by a later request handler", async () => {
		const h = harness();
		await h.start();
		const body = payload("5m");
		await h.request(body);
		body.messages[0]!.content[0]!.cache_control.ttl = "1h";
		await h.settle();
		h.clock.advance(240_000);
		expect(h.compactions).toHaveLength(0);
		h.clock.advance(3_300_000);
		expect(h.compactions).toHaveLength(1);
	});
});

describe("provider-reported cache lifetime", () => {
	const reply = (cacheWrite: number, cacheWrite1h: number | undefined, api = "anthropic-messages") =>
		({ role: "assistant", stopReason: "stop", api, usage: { cacheWrite, cacheWrite1h } });

	test("a one-hour write beats a five-minute payload seen before a later extension rewrote it", async () => {
		const h = harness();
		await h.start();
		await h.request(payload("5m")); // What our observer sees when long-claude runs after it.
		await h.fire("message_end", { message: reply(1200, 1200) });
		await h.settle();
		expect(h.clock.pending).toBe(1);
		h.clock.advance(3_539_999);
		expect(h.compactions).toHaveLength(0);
		h.clock.advance(1);
		expect(h.compactions).toHaveLength(1);
	});

	test.each([
		["a five-minute write", reply(1200, 0), payload("1h")],
		["a mixed 5m/1h write", reply(1200, 800), payload("1h")],
	])("%s stays ineligible even if the payload asked for 1h", async (_name, message, body) => {
		const h = harness({ idleCompactionCacheTtlSeconds: 3600 });
		await h.start();
		await h.request(body);
		await h.fire("message_end", { message });
		await h.settle();
		expect(h.clock.pending).toBe(0);
		expect(h.status()).toContain("short-term cache");
	});

	test.each([
		["a pure cache read", reply(0, 0)],
		["a response without the split", reply(1200, undefined)],
		["another API", reply(1200, 1200, "openai-completions")],
	])("%s gives no evidence, so the payload decides", async (_name, message) => {
		const h = harness();
		await h.start();
		await h.request(payload("5m"));
		await h.fire("message_end", { message });
		await h.settle();
		expect(h.clock.pending).toBe(0);
		expect(h.status()).toContain("short-term cache");
	});

	test("the report applies only to its own request", async () => {
		const h = harness();
		await h.start();
		await h.request(payload("5m"));
		await h.fire("message_end", { message: reply(1200, 1200) });
		await h.request(payload("5m")); // The next request has no report yet.
		await h.fire("message_end", { message: { role: "assistant", stopReason: "stop" } });
		await h.settle();
		expect(h.clock.pending).toBe(0);
	});
});

describe("provider-independent cache lifetimes", () => {
	test.each([
		["anthropic-messages", payload("5m")],
		["anthropic-messages", payload("")],
		["anthropic-messages", { cache_control: { type: "ephemeral" }, messages: [] }],
		["bedrock-converse-stream", { messages: [{ content: [{ cachePoint: { type: "default" } }] }] }],
		["openai-responses", { prompt_cache_retention: "5m" }],
		["openai-responses", { prompt_cache_options: { ttl: "60s" } }],
	] as const)("skips short wire TTLs for %s even with a long configured TTL (%j)", async (api, body) => {
		const h = harness({ idleCompactionCacheTtlSeconds: 3600 });
		h.ctx.model.api = api;
		await h.start();
		await h.request(body);
		await h.settle();
		expect(h.clock.pending).toBe(0);
		expect(h.status()).toContain("short-term cache");
		expect(h.notices).toHaveLength(0); // Ineligible, not misconfigured.
		expect(await h.fire("cache_warming_decision")).toBeUndefined();
		h.clock.advance(3_600_000);
		expect(h.compactions).toHaveLength(0);
	});

	test.each([60, 299, 300, 301])("enforces the five-minute cutoff for metadata/configured TTL %i", async (seconds) => {
		for (const source of ["metadata", "configured"]) {
			const h = harness(source === "configured" ? { idleCompactionCacheTtlSeconds: seconds } : {});
			h.ctx.model.api = "custom-local-api";
			if (source === "metadata") h.ctx.model.promptCache.short = seconds;
			await h.start();
			await h.request({});
			await h.settle();
			expect(h.clock.pending).toBe(seconds > 300 ? 1 : 0);
			if (seconds <= 300) expect(h.status()).toContain("short-term cache");
			h.clock.advance(3_600_000);
			expect(h.compactions).toHaveLength(seconds > 300 ? 1 : 0);
		}
	});

	test("reports the real Codex missing-TTL case instead of implying that enabled means armed", async () => {
		const h = harness();
		h.ctx.model.api = "openai-codex-responses";
		h.ctx.model.provider = "openai-codex";
		h.ctx.model.id = "gpt-6.1-sol";
		h.ctx.contextWindow = 272_000;
		h.ctx.tokens = 192_077; // Above the 50% threshold, but this catalog entry has no promptCache.
		await h.start();
		await h.request({ model: "gpt-6.1-sol", input: [], prompt_cache_key: "test-session", stream: true, store: false });
		await h.settle();
		expect(h.clock.pending).toBe(0);
		expect(h.status()).toContain("not armed: no usable cache TTL");
		expect(h.notices).toHaveLength(1);
		expect(h.notices[0]).toContain("idleCompactionCacheTtlSeconds");
		await h.settle();
		expect(h.notices).toHaveLength(1); // Warn once, not once per turn/status check.
	});

	test("shows the actual deadline and fires at 29 minutes with a configured 30-minute Codex TTL", async () => {
		const h = harness({ idleCompactionCacheTtlSeconds: 1800 });
		h.ctx.model.api = "openai-codex-responses";
		h.ctx.model.provider = "openai-codex";
		h.ctx.model.id = "gpt-6.1-sol";
		h.ctx.contextWindow = 272_000;
		h.ctx.tokens = 192_077;
		await h.start();
		await h.request({ model: "gpt-6.1-sol", input: [], prompt_cache_key: "test-session", stream: true, store: false });
		await h.settle();
		expect(h.status()).toBe(`armed for ${new Date(h.clock.now() + 1_740_000).toISOString()}`);
		h.clock.advance(1_739_999);
		expect(h.compactions).toHaveLength(0);
		h.clock.advance(1);
		expect(h.compactions).toHaveLength(1);
	});

	test("explains a below-threshold session without warning about configuration", async () => {
		const h = harness();
		h.ctx.tokens = 100_000;
		await h.start();
		await h.request();
		await h.settle();
		expect(h.status()).toContain("context usage 10.0% is below 50%");
		expect(h.notices).toHaveLength(0);
	});

	test.each([
		"openai-responses",
		"openai-codex-responses",
		"google-generative-ai",
		"bedrock-converse-stream",
		"mistral-conversations",
		"custom-local-api",
	])("uses model metadata for %s", async (api) => {
		const h = harness();
		h.ctx.model.api = api;
		h.ctx.model.provider = "any-provider";
		h.ctx.model.promptCache.short = 600;
		await h.start();
		await h.request({ messages: [] });
		await h.settle();
		h.clock.advance(539_999);
		expect(h.compactions).toHaveLength(0);
		h.clock.advance(1);
		expect(h.compactions).toHaveLength(1);
	});

	test.each([
		["openai-responses", { prompt_cache_retention: "24h" }, 86_340_000],
		["openai-responses", { prompt_cache_options: { ttl: "30m" } }, 1_740_000],
		["openai-completions", payload("1h"), 3_540_000],
		[
			"bedrock-converse-stream",
			{ messages: [{ content: [{ cachePoint: { type: "default", ttl: "1h" } }] }] },
			3_540_000,
		],
	] as const)("prefers wire TTLs for %s (%j)", async (api, body, delay) => {
		const h = harness({ idleCompactionCacheTtlSeconds: 1200 });
		h.ctx.model.api = api;
		h.ctx.model.promptCache.short = 300;
		await h.start();
		await h.request(body);
		await h.settle();
		h.clock.advance(delay - 1);
		expect(h.compactions).toHaveLength(0);
		h.clock.advance(1);
		expect(h.compactions).toHaveLength(1);
	});

	test("supports a custom provider without HTTP hooks or model metadata using configured TTL", async () => {
		const h = harness({ idleCompactionCacheTtlSeconds: 1200 });
		h.ctx.model.api = "custom-local-api";
		h.ctx.model.provider = "local";
		await h.start();
		await h.fire("agent_start");
		await h.fire("context");
		await h.fire("message_end", { message: { role: "assistant", stopReason: "stop" } });
		await h.settle();
		h.clock.advance(1_140_000);
		expect(h.compactions).toHaveLength(1);
	});

	test("uses long-tier metadata when the request selects it", async () => {
		const h = harness();
		h.ctx.model.api = "pi-messages";
		h.ctx.model.promptCache = { short: 300, long: 3600 };
		await h.start();
		await h.request({ options: { cacheRetention: "long" } });
		await h.settle();
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(1);
	});

	test("does not invent a cache lifetime or override explicit cache disablement", async () => {
		for (const body of [
			{},
			{ options: { cacheRetention: "none" } },
			{ prompt_cache_options: { mode: "explicit" } },
			{ prompt_cache_retention: "none" },
			{ prompt_cache_retention: "unknown" },
		]) {
			const h = harness();
			h.ctx.model.api = "custom-api";
			if (Object.keys(body).length) {
				h.ctx.model.promptCache.short = 600;
				h.settings.idleCompactionCacheTtlSeconds = 1200;
			}
			await h.start();
			await h.request(body);
			await h.settle();
			expect(h.clock.pending).toBe(0);
		}
	});

	test("configured TTL overrides metadata for an otherwise opaque backend", async () => {
		const h = harness({ idleCompactionCacheTtlSeconds: 1200 });
		h.ctx.model.api = "custom-api";
		h.ctx.model.promptCache.short = 300;
		await h.start();
		await h.request({});
		await h.settle();
		h.clock.advance(1_140_000);
		expect(h.compactions).toHaveLength(1);
	});

	test.each([
		{ max_output_tokens: 1 },
		{ max_completion_tokens: 1 },
		{ config: { maxOutputTokens: 1 } },
		{ generationConfig: { maxOutputTokens: 1 } },
		{ inferenceConfig: { maxTokens: 1 } },
		{ options: { maxTokens: 1 } },
	])("ignores provider-specific warming caps %j", async (warm) => {
		const h = harness();
		h.ctx.model.api = "custom-api";
		h.ctx.model.promptCache.short = 600;
		await h.start();
		await h.fire("agent_start");
		await h.fire("context");
		await h.fire("before_provider_request", { payload: { ...warm, prompt_cache_retention: "24h" } });
		await h.fire("before_provider_request", { payload: {} });
		await h.fire("after_provider_response", { status: 200 });
		await h.settle();
		h.clock.advance(540_000);
		expect(h.compactions).toHaveLength(1);
	});
});

describe("eligibility and live configuration", () => {
	test("does not arm or suppress warming when Pi auto-compaction is disabled", async () => {
		const h = harness();
		h.ctx.autoCompactionEnabled = false;
		await h.start();
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(0);
		expect(h.status()).toContain("Pi auto-compaction is disabled");
		expect(await h.fire("cache_warming_decision")).toBeUndefined();
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(0);
		h.ctx.autoCompactionEnabled = true;
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(1);
	});

	test("rechecks Pi auto-compaction and cache TTL before firing an armed timer", async () => {
		for (const change of ["autoCompaction", "ttl"]) {
			const h = harness();
			const body = payload();
			await h.start();
			await h.request(body);
			await h.settle();
			expect(h.clock.pending).toBe(1);
			if (change === "autoCompaction") h.ctx.autoCompactionEnabled = false;
			else body.messages[0]!.content[0]!.cache_control.ttl = "5m";
			expect(await h.fire("cache_warming_decision")).toBeUndefined();
			h.clock.advance(3_540_000);
			expect(h.compactions).toHaveLength(0);
			expect(h.clock.pending).toBe(0);
		}
	});

	test.each([undefined, false, "true", 1])("requires explicit boolean opt-in, not %j", async (enabled) => {
		const h = harness({ idleCompaction: enabled as boolean | undefined });
		await h.start();
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(0);
		expect(await h.fire("cache_warming_decision")).toBeUndefined();
	});

	test.each(["print", "json", "rpc"] as const)("never schedules in %s mode, even with UI", async (mode) => {
		const h = harness({ mode });
		await h.start();
		await h.request();
		await h.settle();
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(0);
		expect(h.clock.pending).toBe(0);
	});

	test("requires a successful request and a known cache lifetime", async () => {
		for (const body of [
			undefined,
			{},
			{ messages: [] },
			{ system: [{ cache_control: { type: "ephemeral" } }], messages: [] },
			payload("24h"),
		]) {
			const h = harness();
			await h.start();
			if (body) await h.request(body);
			await h.settle();
			expect(h.clock.pending).toBe(0);
		}
		const h = harness();
		await h.start();
		await h.request(payload(), 500);
		await h.settle();
		expect(h.clock.pending).toBe(0);
	});

	test.each([200_000, 1_000_000, 301])("defaults to half of a %i-token context window", async (contextWindow) => {
		const h = harness();
		h.ctx.contextWindow = contextWindow;
		h.ctx.tokens = Math.ceil(contextWindow * 0.5) - 1;
		await h.start();
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(0);
		h.ctx.tokens++;
		await h.settle();
		expect(h.clock.pending).toBe(1);
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(1);
	});

	test.each([0, 0.25, 0.333, 1])("accepts a context fraction of %f, including endpoints", async (threshold) => {
		const h = harness({ idleCompactionThreshold: threshold });
		h.ctx.tokens = Math.ceil(h.ctx.contextWindow * threshold);
		await h.start();
		await h.request();
		await h.settle();
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(1);
	});

	test.each([null, "0.5", true, -Infinity])("rejects nonnumeric or nonfinite fractions %j", async (threshold) => {
		const h = harness({ idleCompactionThreshold: threshold as unknown as number });
		await h.start();
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(0);
	});

	test.each([0, -1, NaN, Infinity])("skips an invalid context window of %j", async (contextWindow) => {
		const h = harness({ idleCompactionThreshold: 0 });
		h.ctx.contextWindow = contextWindow;
		await h.start();
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(0);
	});

	test.each([null, 499_999])("does not schedule with %s context tokens", async (tokens) => {
		const h = harness();
		h.ctx.tokens = tokens;
		await h.start();
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(0);
	});

	test.each([
		{ enabled: false },
		{ idleCompaction: false },
		{ idleCompactionThreshold: -0.01 },
		{ idleCompactionThreshold: 1.01 },
		{ idleCompactionThreshold: NaN },
		{ idleCompactionThreshold: Infinity },
		{ idleCompactionBeforeExpirySeconds: 0 },
		{ idleCompactionBeforeExpirySeconds: 3600 },
		{ idleCompactionCacheTtlSeconds: -1 },
		{ idleCompactionCacheTtlSeconds: 1.5 },
		{ idleCompactionCacheTtlSeconds: 3_000_000 },
	])("does not schedule with disabled or unsafe settings %j", async (settings) => {
		const h = harness(settings);
		await h.start();
		await h.request();
		await h.settle();
		expect(h.clock.pending).toBe(0);
	});

	test("rechecks configuration and token count before executing", async () => {
		for (const change of ["enabled", "tokens", "idle", "threshold", "contextWindow"]) {
			const h = harness();
			await h.start();
			await h.request();
			await h.settle();
			if (change === "enabled") h.settings.enabled = false;
			if (change === "tokens") h.ctx.tokens = null;
			if (change === "idle") h.settings.idleCompaction = false;
			if (change === "threshold") h.settings.idleCompactionThreshold = 0.75;
			if (change === "contextWindow") h.ctx.contextWindow = 2_000_000;
			h.clock.advance(3_540_000);
			expect(h.compactions).toHaveLength(0);
		}
	});

	test("custom thresholds and live refresh take effect without resetting cache age", async () => {
		const h = harness({ idleCompactionThreshold: 0.1, idleCompactionBeforeExpirySeconds: 120 });
		h.ctx.tokens = 100_000;
		await h.start();
		await h.request();
		await h.settle();
		h.clock.advance(1_000_000);
		h.settings.idleCompaction = false;
		h.refresh();
		expect(h.clock.pending).toBe(0);
		h.settings.idleCompaction = true;
		h.refresh();
		h.clock.advance(2_480_000);
		expect(h.compactions).toHaveLength(1);
	});
});

describe("lifecycle safety and cache warming", () => {
	test("a child session sharing the process cannot cancel or rearm its parent's timer", async () => {
		const h = harness();
		await h.start();
		await h.request();
		await h.settle();
		const child = { ...h.ctx, mode: "print" as const, sessionManager: { getSessionId: () => "child" } };
		for (const event of [
			"session_start",
			"agent_start",
			"context",
			"before_provider_request",
			"after_provider_response",
			"agent_settled",
			"session_shutdown",
		]) {
			await h.fire(event, { payload: payload(), status: 200 }, child);
		}
		expect(h.clock.pending).toBe(1);
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(1);
	});

	test.each([
		"input",
		"agent_start",
		"context",
		"model_select",
		"thinking_level_select",
		"session_before_tree",
		"session_tree",
		"session_before_switch",
		"session_before_compact",
		"session_compact",
		"session_compact_failed",
		"session_shutdown",
	])("invalidates pending work on %s", async (event) => {
		const h = harness();
		await h.start();
		await h.request();
		await h.settle();
		await h.fire(event);
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(0);
		expect(h.clock.pending).toBe(0);
	});

	test.each(["busy", "pending", "editor", "model", "thinking"])(
		"does not interrupt %s at the deadline",
		async (change) => {
			const h = harness();
			await h.start();
			await h.request();
			await h.settle();
			if (change === "busy") h.ctx.busy = true;
			if (change === "pending") h.ctx.pending = true;
			if (change === "editor") h.ctx.editor = "Draft user message";
			if (change === "model") h.ctx.model.id = "other-claude";
			if (change === "thinking") h.ctx.thinkingLevel = "off";
			h.clock.advance(3_540_000);
			expect(h.compactions).toHaveLength(0);
		},
	);

	test("waits for nested UI prompts to close", async () => {
		const h = harness();
		await h.start();
		await h.request();
		await h.settle();
		await h.fire("ui_prompt_start");
		await h.fire("ui_prompt_start");
		h.clock.advance(3_550_000);
		await h.fire("ui_prompt_end");
		h.clock.advance(0);
		expect(h.compactions).toHaveLength(0);
		await h.fire("ui_prompt_end");
		h.clock.advance(0);
		expect(h.compactions).toHaveLength(1);
	});

	test("new ordinary requests replace the old deadline", async () => {
		const h = harness();
		await h.start();
		await h.request();
		await h.settle();
		h.clock.advance(1_000_000);
		await h.request();
		await h.settle();
		h.clock.advance(2_540_000);
		expect(h.compactions).toHaveLength(0);
		h.clock.advance(1_000_000);
		expect(h.compactions).toHaveLength(1);
	});

	test("leaves active warming alone, stops eligible idle warming, and ignores replayed payloads", async () => {
		const h = harness();
		await h.start();
		await h.request();
		h.ctx.busy = true;
		expect(await h.fire("cache_warming_decision")).toBeUndefined();
		h.ctx.busy = false;
		await h.settle();
		expect(await h.fire("cache_warming_decision")).toEqual({ action: "stop" });
		h.clock.advance(1_000_000);
		await h.fire("before_provider_request", { payload: payload() }); // No context event: warming replay.
		await h.fire("after_provider_response", { status: 200 });
		await h.settle();
		h.clock.advance(2_540_000);
		expect(h.compactions).toHaveLength(1); // Not postponed by the warm request.
		await h.fire("before_provider_request", { payload: payload() });
		await h.settle();
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(1); // Nor rearmed after compaction.
	});

	test("ignores a one-token warming replay racing with preparation of the next request", async () => {
		const h = harness();
		await h.start();
		await h.fire("agent_start");
		await h.fire("context");
		await h.fire("before_provider_request", { payload: { ...payload("5m"), max_tokens: 1 } });
		await h.fire("after_provider_response", { status: 200 });
		await h.fire("before_provider_request", { payload: { ...payload("1h"), max_tokens: 8192 } });
		await h.fire("after_provider_response", { status: 200 });
		await h.settle();
		h.clock.advance(240_000);
		expect(h.compactions).toHaveLength(0);
		h.clock.advance(3_300_000);
		expect(h.compactions).toHaveLength(1);
	});

	test("does not stop warming after a draft caused the compaction timer to be skipped", async () => {
		const h = harness();
		await h.start();
		await h.request(payload());
		await h.settle();
		expect(h.clock.pending).toBe(1);
		h.ctx.editor = "Draft";
		h.clock.advance(3_540_000);
		h.ctx.editor = "";
		expect(h.clock.pending).toBe(0);
		expect(await h.fire("cache_warming_decision")).toBeUndefined();
	});

	test("does not retry failed compaction until a new ordinary request", async () => {
		const h = harness();
		await h.start();
		await h.request();
		await h.settle();
		h.clock.advance(3_540_000);
		h.compactions[0]!.onError!(new Error("test failure"));
		expect(h.notices[0]).toContain("test failure");
		await h.settle();
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(1);
		await h.request();
		await h.settle();
		h.clock.advance(3_540_000);
		expect(h.compactions).toHaveLength(2);
	});

	test("does not mutate the payload or process environment", async () => {
		const h = harness();
		const env = process.env.PI_CACHE_RETENTION;
		const body = payload();
		const original = JSON.stringify(body);
		await h.start();
		await h.request(body);
		await h.settle();
		h.clock.advance(3_540_000);
		expect(JSON.stringify(body)).toBe(original);
		expect(process.env.PI_CACHE_RETENTION).toBe(env);
	});
});
