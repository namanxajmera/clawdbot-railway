import fs from "node:fs/promises";
import os from "node:os";
import { createOpenAI } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type {
  AgentMessage,
  AgentToolResult,
} from "@mariozechner/pi-agent-core";
import type {
  Api,
  AssistantMessage,
  ImageContent,
  Message,
  Model,
  ToolCall as PiToolCall,
  TextContent,
  ThinkingContent,
  ToolResultMessage,
  Usage,
  UserMessage,
} from "@mariozechner/pi-ai";
import {
  discoverAuthStorage,
  discoverModels,
  SessionManager,
} from "@mariozechner/pi-coding-agent";
import {
  type AssistantModelMessage,
  jsonSchema,
  type ModelMessage,
  streamText,
  type ToolCallPart,
  type ToolModelMessage,
  type ToolResultPart,
  type ToolSet,
  tool,
} from "ai";

import { resolveHeartbeatPrompt } from "../auto-reply/heartbeat.js";
import { parseReplyDirectives } from "../auto-reply/reply/reply-directives.js";
import type { ReasoningLevel, ThinkLevel } from "../auto-reply/thinking.js";
import { formatToolAggregate } from "../auto-reply/tool-meta.js";
import type { ClawdbotConfig } from "../config/config.js";
import { resolveProviderCapabilities } from "../config/provider-capabilities.js";
import { emitAgentEvent } from "../infra/agent-events.js";
import { getMachineDisplayName } from "../infra/machine-name.js";
import { createSubsystemLogger } from "../logging.js";
import { enqueueCommandInLane } from "../process/command-queue.js";
import {
  getProviderPlugin,
  normalizeProviderId,
} from "../providers/plugins/index.js";
import { normalizeMessageProvider } from "../utils/message-provider.js";
import { isReasoningTagProvider } from "../utils/provider-utils.js";
import { resolveUserPath } from "../utils.js";
import { resolveClawdbotAgentDir } from "./agent-paths.js";
import { resolveSessionAgentIds } from "./agent-scope.js";
import type { ExecElevatedDefaults, ExecToolDefaults } from "./bash-tools.js";
import {
  DEFAULT_CONTEXT_TOKENS,
  DEFAULT_MODEL,
  DEFAULT_PROVIDER,
} from "./defaults.js";
import {
  registerEmbeddedRun,
  unregisterEmbeddedRun,
} from "./embedded-run-state.js";
import { getApiKeyForModel, resolveModelAuthMode } from "./model-auth.js";
import { normalizeModelCompat } from "./model-compat.js";
import { ensureClawdbotModelsJson } from "./models-config.js";
import { EmbeddedBlockChunker } from "./pi-embedded-block-chunker.js";
import {
  buildBootstrapContextFiles,
  ensureSessionHeader,
  isContextOverflowError,
  isMessagingToolDuplicateNormalized,
  normalizeTextForComparison,
  sanitizeSessionMessagesImages,
} from "./pi-embedded-helpers.js";
import type { MessagingToolSend } from "./pi-embedded-messaging.js";
import {
  isMessagingTool,
  isMessagingToolSendAction,
  normalizeTargetForProvider,
} from "./pi-embedded-messaging.js";
import type {
  EmbeddedPiAgentMeta,
  EmbeddedPiRunResult,
  runEmbeddedPiAgent,
} from "./pi-embedded-runner.js";
import {
  extractAssistantText,
  extractAssistantThinking,
  extractThinkingFromTaggedStream,
  formatReasoningMessage,
  inferToolMetaFromArgs,
} from "./pi-embedded-utils.js";
import { createClawdbotCodingTools } from "./pi-tools.js";
import { resolveSandboxContext } from "./sandbox.js";
import { guardSessionManager } from "./session-tool-result-guard-wrapper.js";
import { sanitizeToolUseResultPairing } from "./session-transcript-repair.js";
import { acquireSessionWriteLock } from "./session-write-lock.js";
import {
  applySkillEnvOverrides,
  applySkillEnvOverridesFromSnapshot,
  loadWorkspaceSkillEntries,
  resolveSkillsPromptForRun,
  type SkillSnapshot,
} from "./skills.js";
import { buildAgentSystemPrompt } from "./system-prompt.js";
import { buildToolSummaryMap } from "./tool-summaries.js";
import { normalizeUsage, type UsageLike } from "./usage.js";
import {
  filterBootstrapFilesForSession,
  loadWorkspaceBootstrapFiles,
} from "./workspace.js";

const OPENAI_COMPAT_APIS = new Set<Api>([
  "openai-completions",
  "openai-responses",
  "openai-codex-responses",
]);

const THINKING_TAG_SCAN_RE =
  /<\s*(\/?)\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
const FINAL_TAG_SCAN_RE = /<\s*(\/?)\s*final\s*>/gi;

const log = createSubsystemLogger("agent/embedded");

type EmbeddedAgentRunParams = Parameters<typeof runEmbeddedPiAgent>[0];
type PiModel = Model<Api>;

type EmbeddedSandboxInfo = {
  enabled: boolean;
  workspaceDir?: string;
  workspaceAccess?: "none" | "ro" | "rw";
  agentWorkspaceMount?: string;
  browserControlUrl?: string;
  browserNoVncUrl?: string;
  hostBrowserAllowed?: boolean;
  allowedControlUrls?: string[];
  allowedControlHosts?: string[];
  allowedControlPorts?: number[];
  elevated?: {
    allowed: boolean;
    defaultLevel: "on" | "off";
  };
};

type BlockTagState = { thinking: boolean; final: boolean };

function resolveSessionLane(key: string) {
  const cleaned = key.trim() || "main";
  return cleaned.startsWith("session:") ? cleaned : `session:${cleaned}`;
}

function resolveGlobalLane(lane?: string) {
  const cleaned = lane?.trim();
  return cleaned ? cleaned : "main";
}

function resolveExecToolDefaults(
  config?: ClawdbotConfig,
): ExecToolDefaults | undefined {
  const tools = config?.tools;
  if (!tools) return undefined;
  if (!tools.exec) return tools.bash;
  if (!tools.bash) return tools.exec;
  return { ...tools.bash, ...tools.exec };
}

function resolveUserTimezone(configured?: string): string {
  const trimmed = configured?.trim();
  if (trimmed) {
    try {
      new Intl.DateTimeFormat("en-US", { timeZone: trimmed }).format(
        new Date(),
      );
      return trimmed;
    } catch {
      // ignore invalid timezone
    }
  }
  const host = Intl.DateTimeFormat().resolvedOptions().timeZone;
  return host?.trim() || "UTC";
}

function formatUserTime(date: Date, timeZone: string): string | undefined {
  try {
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone,
      weekday: "long",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23",
    }).formatToParts(date);
    const map: Record<string, string> = {};
    for (const part of parts) {
      if (part.type !== "literal") map[part.type] = part.value;
    }
    if (
      !map.weekday ||
      !map.year ||
      !map.month ||
      !map.day ||
      !map.hour ||
      !map.minute
    )
      return undefined;
    return `${map.weekday}, ${map.year}-${map.month}-${map.day} ${map.hour}:${map.minute}`;
  } catch {
    return undefined;
  }
}

function describeUnknownError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  try {
    const serialized = JSON.stringify(error);
    return serialized ?? "Unknown error";
  } catch {
    return "Unknown error";
  }
}

function buildEmbeddedSandboxInfo(
  sandbox?: Awaited<ReturnType<typeof resolveSandboxContext>>,
  execElevated?: ExecElevatedDefaults,
): EmbeddedSandboxInfo | undefined {
  if (!sandbox?.enabled) return undefined;
  const elevatedAllowed = Boolean(
    execElevated?.enabled && execElevated.allowed,
  );
  return {
    enabled: true,
    workspaceDir: sandbox.workspaceDir,
    workspaceAccess: sandbox.workspaceAccess,
    agentWorkspaceMount:
      sandbox.workspaceAccess === "ro" ? "/agent" : undefined,
    browserControlUrl: sandbox.browser?.controlUrl,
    browserNoVncUrl: sandbox.browser?.noVncUrl,
    hostBrowserAllowed: sandbox.browserAllowHostControl,
    allowedControlUrls: sandbox.browserAllowedControlUrls,
    allowedControlHosts: sandbox.browserAllowedControlHosts,
    allowedControlPorts: sandbox.browserAllowedControlPorts,
    ...(elevatedAllowed
      ? {
          elevated: {
            allowed: true,
            defaultLevel: execElevated?.defaultLevel ?? "off",
          },
        }
      : {}),
  };
}

function resolveModel(
  provider: string,
  modelId: string,
  agentDir?: string,
  cfg?: ClawdbotConfig,
): {
  model?: PiModel;
  error?: string;
  authStorage: ReturnType<typeof discoverAuthStorage>;
  modelRegistry: ReturnType<typeof discoverModels>;
} {
  const resolvedAgentDir = agentDir ?? resolveClawdbotAgentDir();
  const authStorage = discoverAuthStorage(resolvedAgentDir);
  const modelRegistry = discoverModels(authStorage, resolvedAgentDir);
  const model = modelRegistry.find(provider, modelId) as PiModel | null;
  if (!model) {
    const providers = cfg?.models?.providers ?? {};
    const inlineModels =
      providers[provider]?.models ??
      Object.values(providers)
        .flatMap((entry) => entry?.models ?? [])
        .map((entry) => ({ ...entry, provider }));
    const inlineMatch = inlineModels.find((entry) => entry.id === modelId);
    if (inlineMatch) {
      const normalized = normalizeModelCompat(inlineMatch as PiModel);
      return {
        model: normalized,
        authStorage,
        modelRegistry,
      };
    }
    const providerCfg = providers[provider];
    if (providerCfg || modelId.startsWith("mock-")) {
      const fallbackModel: PiModel = normalizeModelCompat({
        id: modelId,
        name: modelId,
        api: providerCfg?.api ?? "openai-responses",
        provider,
        baseUrl: providerCfg?.baseUrl ?? "",
        headers: providerCfg?.headers ?? undefined,
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow:
          providerCfg?.models?.[0]?.contextWindow ?? DEFAULT_CONTEXT_TOKENS,
        maxTokens:
          providerCfg?.models?.[0]?.maxTokens ?? DEFAULT_CONTEXT_TOKENS,
      } as PiModel);
      return { model: fallbackModel, authStorage, modelRegistry };
    }
    return {
      error: `Unknown model: ${provider}/${modelId}`,
      authStorage,
      modelRegistry,
    };
  }
  return { model: normalizeModelCompat(model), authStorage, modelRegistry };
}

function resolveExtraParams(params: {
  cfg: ClawdbotConfig | undefined;
  provider: string;
  modelId: string;
}): { temperature?: number; maxOutputTokens?: number } {
  const modelKey = `${params.provider}/${params.modelId}`;
  const modelConfig = params.cfg?.agents?.defaults?.models?.[modelKey];
  const extraParams = modelConfig?.params ?? {};
  const temperature =
    typeof extraParams.temperature === "number"
      ? extraParams.temperature
      : undefined;
  const maxOutputTokens =
    typeof extraParams.maxTokens === "number"
      ? extraParams.maxTokens
      : undefined;
  return { temperature, maxOutputTokens };
}

function limitHistoryTurns(
  messages: AgentMessage[],
  limit: number | undefined,
): AgentMessage[] {
  if (!limit || limit <= 0 || messages.length === 0) return messages;

  let userCount = 0;
  let lastUserIndex = messages.length;

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "user") {
      userCount += 1;
      if (userCount > limit) {
        return messages.slice(lastUserIndex);
      }
      lastUserIndex = i;
    }
  }
  return messages;
}

function getDmHistoryLimitFromSessionKey(
  sessionKey: string | undefined,
  config: ClawdbotConfig | undefined,
): number | undefined {
  if (!sessionKey || !config) return undefined;

  const sessionKeyLower = sessionKey.toLowerCase();
  const sessionKeyParts = sessionKeyLower.split(":");

  // Support both "provider:dm:userId" and "agent:...:provider:dm:userId" forms
  const providerIndex = sessionKeyParts[0] === "agent" ? 2 : 0;
  const provider = sessionKeyParts[providerIndex];
  const kind = sessionKeyParts[providerIndex + 1];
  const userId = sessionKeyParts.slice(providerIndex + 2).join(":");

  if (kind !== "dm") return undefined;

  const getLimit = (
    providerConfig:
      | {
          dmHistoryLimit?: number;
          dms?: Record<string, { historyLimit?: number }>;
        }
      | undefined,
  ): number | undefined => {
    if (!providerConfig) return undefined;
    if (
      userId &&
      kind === "dm" &&
      providerConfig.dms?.[userId]?.historyLimit !== undefined
    ) {
      return providerConfig.dms[userId].historyLimit;
    }
    return providerConfig.dmHistoryLimit;
  };

  switch (provider) {
    case "telegram":
      return getLimit(config.telegram);
    case "whatsapp":
      return getLimit(config.whatsapp);
    case "discord":
      return getLimit(config.discord);
    case "slack":
      return getLimit(config.slack);
    case "signal":
      return getLimit(config.signal);
    case "imessage":
      return getLimit(config.imessage);
    case "msteams":
      return getLimit(config.msteams);
    default:
      return undefined;
  }
}

function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const name = "name" in err ? String(err.name) : "";
  if (name === "AbortError") return true;
  const message =
    "message" in err && typeof err.message === "string"
      ? err.message.toLowerCase()
      : "";
  return message.includes("aborted");
}

function stripBlockTags(
  text: string,
  state: BlockTagState,
  enforceFinalTag: boolean | undefined,
): string {
  if (!text) return text;

  let processed = "";
  THINKING_TAG_SCAN_RE.lastIndex = 0;
  let lastIndex = 0;
  let inThinking = state.thinking;
  for (const match of text.matchAll(THINKING_TAG_SCAN_RE)) {
    const idx = match.index ?? 0;
    if (!inThinking) {
      processed += text.slice(lastIndex, idx);
    }
    const isClose = match[1] === "/";
    inThinking = !isClose;
    lastIndex = idx + match[0].length;
  }
  if (!inThinking) {
    processed += text.slice(lastIndex);
  }
  state.thinking = inThinking;

  if (!enforceFinalTag) {
    FINAL_TAG_SCAN_RE.lastIndex = 0;
    return processed.replace(FINAL_TAG_SCAN_RE, "");
  }

  let result = "";
  FINAL_TAG_SCAN_RE.lastIndex = 0;
  let lastFinalIndex = 0;
  let inFinal = state.final;
  let everInFinal = state.final;

  for (const match of processed.matchAll(FINAL_TAG_SCAN_RE)) {
    const idx = match.index ?? 0;
    const isClose = match[1] === "/";

    if (!inFinal && !isClose) {
      inFinal = true;
      everInFinal = true;
      lastFinalIndex = idx + match[0].length;
    } else if (inFinal && isClose) {
      result += processed.slice(lastFinalIndex, idx);
      inFinal = false;
      lastFinalIndex = idx + match[0].length;
    }
  }

  if (inFinal) {
    result += processed.slice(lastFinalIndex);
  }
  state.final = inFinal;

  if (!everInFinal) {
    return "";
  }

  return result.replace(FINAL_TAG_SCAN_RE, "");
}

function isToolResultError(result: unknown): boolean {
  if (!result || typeof result !== "object") return false;
  const record = result as { details?: unknown };
  const details = record.details;
  if (!details || typeof details !== "object") return false;
  const status = (details as { status?: unknown }).status;
  if (typeof status !== "string") return false;
  const normalized = status.trim().toLowerCase();
  return normalized === "error" || normalized === "timeout";
}

function extractMessagingToolSend(
  toolName: string,
  args: Record<string, unknown>,
): MessagingToolSend | undefined {
  const action = typeof args.action === "string" ? args.action.trim() : "";
  const accountIdRaw =
    typeof args.accountId === "string" ? args.accountId.trim() : undefined;
  const accountId = accountIdRaw ? accountIdRaw : undefined;
  if (toolName === "message") {
    if (action !== "send" && action !== "thread-reply") return undefined;
    const toRaw = typeof args.to === "string" ? args.to : undefined;
    if (!toRaw) return undefined;
    const providerRaw =
      typeof args.provider === "string" ? args.provider.trim() : "";
    const providerId = providerRaw ? normalizeProviderId(providerRaw) : null;
    const provider =
      providerId ?? (providerRaw ? providerRaw.toLowerCase() : "message");
    const to = normalizeTargetForProvider(provider, toRaw);
    return to ? { tool: toolName, provider, accountId, to } : undefined;
  }
  const providerId = normalizeProviderId(toolName);
  if (!providerId) return undefined;
  const plugin = getProviderPlugin(providerId);
  const extracted = plugin?.actions?.extractToolSend?.({ args });
  if (!extracted?.to) return undefined;
  const to = normalizeTargetForProvider(providerId, extracted.to);
  return to
    ? {
        tool: toolName,
        provider: providerId,
        accountId: extracted.accountId ?? accountId,
        to,
      }
    : undefined;
}

function normalizeUsageFromAi(usage?: UsageLike | null):
  | {
      input?: number;
      output?: number;
      cacheRead?: number;
      cacheWrite?: number;
      total?: number;
    }
  | undefined {
  if (!usage) return undefined;
  return normalizeUsage(usage);
}

function mapUsageToPi(usage?: {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  total?: number;
}): Usage {
  const input = usage?.input ?? 0;
  const output = usage?.output ?? 0;
  const cacheRead = usage?.cacheRead ?? 0;
  const cacheWrite = usage?.cacheWrite ?? 0;
  const total = usage?.total ?? input + output + cacheRead + cacheWrite;
  return {
    input,
    output,
    cacheRead,
    cacheWrite,
    totalTokens: total,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: 0,
    },
  };
}

function buildToolOutput(result: AgentToolResult<unknown>): unknown {
  if (result.details !== undefined) return result.details;
  if (!Array.isArray(result.content)) return {};
  const text = result.content
    .map((block) => {
      if (!block || typeof block !== "object") return "";
      const record = block as { type?: unknown; text?: unknown };
      if (record.type !== "text" || typeof record.text !== "string") return "";
      return record.text.trim();
    })
    .filter(Boolean)
    .join("\n")
    .trim();
  return text ? { text } : {};
}

function convertUserContent(
  content: string | (TextContent | ImageContent)[],
):
  | string
  | Array<
      | { type: "text"; text: string }
      | { type: "image"; image: string; mediaType?: string }
    > {
  if (typeof content === "string") return content;
  const parts: Array<
    | { type: "text"; text: string }
    | { type: "image"; image: string; mediaType?: string }
  > = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ type: "text", text: block.text });
    }
    if (block.type === "image" && typeof block.data === "string") {
      parts.push({
        type: "image",
        image: block.data,
        mediaType: block.mimeType,
      });
    }
  }
  return parts.length === 1 && parts[0].type === "text" ? parts[0].text : parts;
}

function convertAgentMessagesToAi(messages: AgentMessage[]): ModelMessage[] {
  const out: ModelMessage[] = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const role = (msg as { role?: unknown }).role;
    if (role === "user") {
      const userMsg = msg as UserMessage;
      out.push({
        role: "user",
        content: convertUserContent(userMsg.content),
      });
      continue;
    }
    if (role === "assistant") {
      const assistantMsg = msg as AssistantMessage;
      const content = assistantMsg.content;
      if (typeof content === "string") {
        out.push({ role: "assistant", content });
        continue;
      }
      const parts: Array<
        | { type: "text"; text: string }
        | { type: "reasoning"; text: string }
        | {
            type: "tool-call";
            toolCallId: string;
            toolName: string;
            input: unknown;
          }
      > = [];
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!block || typeof block !== "object") continue;
          if (block.type === "text" && typeof block.text === "string") {
            parts.push({ type: "text", text: block.text });
          }
          if (block.type === "thinking" && typeof block.thinking === "string") {
            parts.push({ type: "reasoning", text: block.thinking });
          }
          if (block.type === "toolCall") {
            const toolBlock = block as PiToolCall;
            parts.push({
              type: "tool-call",
              toolCallId: toolBlock.id,
              toolName: toolBlock.name,
              input: toolBlock.arguments,
            });
          }
        }
      }
      out.push({ role: "assistant", content: parts });
      continue;
    }
    if (role === "toolResult") {
      const toolMsg = msg as ToolResultMessage;
      const details = (toolMsg as { details?: unknown }).details;
      const text = Array.isArray(toolMsg.content)
        ? toolMsg.content
            .map((block) => {
              if (!block || typeof block !== "object") return "";
              const record = block as { type?: unknown; text?: unknown };
              if (record.type !== "text" || typeof record.text !== "string") {
                return "";
              }
              return record.text.trim();
            })
            .filter(Boolean)
            .join("\n")
            .trim()
        : "";
      const output = details ? JSON.stringify(details, null, 2) : text;
      const safeOutput = { type: "text", value: output || "" } as const;
      out.push({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: toolMsg.toolCallId,
            toolName: toolMsg.toolName,
            output: safeOutput,
          },
        ],
      });
    }
  }
  return out;
}

function convertAssistantToPi(params: {
  message: AssistantModelMessage;
  model: PiModel;
  usage?: Usage;
  stopReason?: "stop" | "length" | "toolUse" | "error" | "aborted";
}): AssistantMessage {
  const content = params.message.content;
  const blocks: Array<TextContent | ThinkingContent | PiToolCall> = [];

  if (typeof content === "string") {
    blocks.push({ type: "text", text: content });
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (!part || typeof part !== "object") continue;
      if (part.type === "text" && typeof part.text === "string") {
        blocks.push({ type: "text", text: part.text });
      }
      if (part.type === "reasoning" && typeof part.text === "string") {
        blocks.push({ type: "thinking", thinking: part.text });
      }
      if (part.type === "tool-call") {
        const toolPart = part as ToolCallPart;
        blocks.push({
          type: "toolCall",
          id: toolPart.toolCallId,
          name: toolPart.toolName,
          arguments: toolPart.input as Record<string, unknown>,
        });
      }
    }
  }

  return {
    role: "assistant",
    content: blocks,
    api: params.model.api,
    provider: params.model.provider,
    model: params.model.id,
    usage: params.usage ?? mapUsageToPi(undefined),
    stopReason: params.stopReason ?? "stop",
    timestamp: Date.now(),
  };
}

function convertToolResultToPi(
  part: ToolResultPart,
): ToolResultMessage<unknown> {
  let text = "";
  let isError = false;
  const output = part.output;
  if (output.type === "text" && typeof output.value === "string") {
    text = output.value;
  } else if (output.type === "json") {
    text = JSON.stringify(output.value, null, 2);
  } else if (output.type === "error-text" && typeof output.value === "string") {
    text = output.value;
    isError = true;
  } else if (output.type === "error-json") {
    text = JSON.stringify(output.value, null, 2);
    isError = true;
  } else if (output.type === "execution-denied") {
    text = output.reason ?? "Tool execution denied.";
    isError = true;
  }

  return {
    role: "toolResult",
    toolCallId: part.toolCallId,
    toolName: part.toolName,
    content: text ? [{ type: "text", text }] : [],
    details: output.type === "json" ? output.value : undefined,
    isError,
    timestamp: Date.now(),
  };
}

function resolveFinishReason(
  reason?: string,
): "stop" | "length" | "toolUse" | "error" | "aborted" {
  if (!reason) return "stop";
  const normalized = reason.toLowerCase();
  if (normalized === "length") return "length";
  if (
    normalized === "tool-calls" ||
    normalized === "tool_calls" ||
    normalized === "tool-call"
  )
    return "toolUse";
  if (normalized === "content-filter") return "error";
  if (normalized === "abort") return "aborted";
  if (normalized === "error") return "error";
  return "stop";
}

export async function runEmbeddedVercelAgent(
  params: EmbeddedAgentRunParams,
): Promise<EmbeddedPiRunResult> {
  const sessionLane = resolveSessionLane(
    params.sessionKey?.trim() || params.sessionId,
  );
  const globalLane = resolveGlobalLane(params.lane);
  const enqueueGlobal =
    params.enqueue ??
    ((task, opts) => enqueueCommandInLane(globalLane, task, opts));
  const runAbortController = new AbortController();

  return enqueueCommandInLane(sessionLane, () =>
    enqueueGlobal(async () => {
      const started = Date.now();
      const resolvedWorkspace = resolveUserPath(params.workspaceDir);
      const prevCwd = process.cwd();

      const provider =
        (params.provider ?? DEFAULT_PROVIDER).trim() || DEFAULT_PROVIDER;
      const modelId = (params.model ?? DEFAULT_MODEL).trim() || DEFAULT_MODEL;
      const agentDir = params.agentDir ?? resolveClawdbotAgentDir();
      await ensureClawdbotModelsJson(params.config, agentDir);
      const { model, error } = resolveModel(
        provider,
        modelId,
        agentDir,
        params.config,
      );
      if (!model) {
        return {
          payloads: [
            {
              text: error ?? `Unknown model: ${provider}/${modelId}`,
              isError: true,
            },
          ],
          meta: {
            durationMs: Date.now() - started,
            agentMeta: {
              sessionId: params.sessionId,
              provider,
              model: modelId,
            },
          },
        };
      }

      if (!OPENAI_COMPAT_APIS.has(model.api)) {
        return {
          payloads: [
            {
              text:
                `Vercel SDK runtime only supports OpenAI-compatible models today. ` +
                `Current model is ${provider}/${modelId} (${model.api}). ` +
                `Switch agents.defaults.runtime to "pi" or choose an OpenAI-compatible model.`,
              isError: true,
            },
          ],
          meta: {
            durationMs: Date.now() - started,
            agentMeta: {
              sessionId: params.sessionId,
              provider,
              model: model.id,
            },
          },
        };
      }

      let apiKeyInfo: { apiKey: string; profileId?: string; source: string };
      try {
        apiKeyInfo = await getApiKeyForModel({
          model,
          cfg: params.config,
          profileId: params.authProfileId,
          agentDir,
        });
      } catch (err) {
        return {
          payloads: [{ text: describeUnknownError(err), isError: true }],
          meta: {
            durationMs: Date.now() - started,
            agentMeta: {
              sessionId: params.sessionId,
              provider,
              model: model.id,
            },
          },
        };
      }

      let baseUrl = model.baseUrl;
      const headers = model.headers;
      let apiKey = apiKeyInfo.apiKey;

      if (model.provider === "github-copilot") {
        const { resolveCopilotApiToken } = await import(
          "../providers/github-copilot-token.js"
        );
        const copilotToken = await resolveCopilotApiToken({
          githubToken: apiKeyInfo.apiKey,
        });
        apiKey = copilotToken.token;
        baseUrl = copilotToken.baseUrl;
      }

      const extraParams = resolveExtraParams({
        cfg: params.config,
        provider,
        modelId,
      });

      const resolvedBaseUrl = baseUrl?.trim();
      if (!resolvedBaseUrl && model.provider !== "openai") {
        return {
          payloads: [
            {
              text:
                `OpenAI-compatible baseUrl is missing for ${provider}/${modelId}. ` +
                `Set models.providers.${provider}.baseUrl or switch agents.defaults.runtime to "pi".`,
              isError: true,
            },
          ],
          meta: {
            durationMs: Date.now() - started,
            agentMeta: {
              sessionId: params.sessionId,
              provider,
              model: model.id,
            },
          },
        };
      }

      const openaiOptions = {
        apiKey,
        headers,
        ...(resolvedBaseUrl ? { baseURL: resolvedBaseUrl } : {}),
      };

      const languageModel =
        model.provider === "openai"
          ? createOpenAI(openaiOptions)(model.id)
          : createOpenAICompatible({
              apiKey,
              baseURL: resolvedBaseUrl ?? "",
              headers,
              name: model.provider,
            })(model.id);

      await fs.mkdir(resolvedWorkspace, { recursive: true });
      const sandboxSessionKey = params.sessionKey?.trim() || params.sessionId;
      const sandbox = await resolveSandboxContext({
        config: params.config,
        sessionKey: sandboxSessionKey,
        workspaceDir: resolvedWorkspace,
      });
      const effectiveWorkspace = sandbox?.enabled
        ? sandbox.workspaceAccess === "rw"
          ? resolvedWorkspace
          : sandbox.workspaceDir
        : resolvedWorkspace;
      await fs.mkdir(effectiveWorkspace, { recursive: true });

      let restoreSkillEnv: (() => void) | undefined;
      process.chdir(effectiveWorkspace);
      try {
        const shouldLoadSkillEntries =
          !params.skillsSnapshot || !params.skillsSnapshot.resolvedSkills;
        const skillEntries = shouldLoadSkillEntries
          ? loadWorkspaceSkillEntries(effectiveWorkspace)
          : [];
        restoreSkillEnv = params.skillsSnapshot
          ? applySkillEnvOverridesFromSnapshot({
              snapshot: params.skillsSnapshot as SkillSnapshot,
              config: params.config,
            })
          : applySkillEnvOverrides({
              skills: skillEntries ?? [],
              config: params.config,
            });
        const skillsPrompt = resolveSkillsPromptForRun({
          skillsSnapshot: params.skillsSnapshot as SkillSnapshot | undefined,
          entries: shouldLoadSkillEntries ? skillEntries : undefined,
          config: params.config,
          workspaceDir: effectiveWorkspace,
        });

        const bootstrapFiles = filterBootstrapFilesForSession(
          await loadWorkspaceBootstrapFiles(effectiveWorkspace),
          params.sessionKey ?? params.sessionId,
        );
        const contextFiles = buildBootstrapContextFiles(bootstrapFiles);

        const tools = createClawdbotCodingTools({
          exec: {
            ...resolveExecToolDefaults(params.config),
            elevated: params.bashElevated,
          },
          sandbox,
          messageProvider: params.messageProvider,
          agentAccountId: params.agentAccountId,
          sessionKey: params.sessionKey ?? params.sessionId,
          agentDir,
          workspaceDir: effectiveWorkspace,
          config: params.config,
          abortSignal: runAbortController.signal,
          modelProvider: model.provider,
          modelId,
          modelAuthMode: resolveModelAuthMode(model.provider, params.config),
          currentChannelId: params.currentChannelId,
          currentThreadTs: params.currentThreadTs,
          replyToMode: params.replyToMode,
          hasRepliedRef: params.hasRepliedRef,
        });

        const machineName = await getMachineDisplayName();
        const runtimeProvider = normalizeMessageProvider(
          params.messageProvider,
        );
        const runtimeCapabilities = runtimeProvider
          ? (resolveProviderCapabilities({
              cfg: params.config,
              provider: runtimeProvider,
              accountId: params.agentAccountId,
            }) ?? [])
          : undefined;
        const runtimeInfo = {
          host: machineName,
          os: `${os.type()} ${os.release()}`,
          arch: os.arch(),
          node: process.version,
          model: `${provider}/${modelId}`,
          provider: runtimeProvider,
          capabilities: runtimeCapabilities,
        };
        const sandboxInfo = buildEmbeddedSandboxInfo(
          sandbox,
          params.bashElevated,
        );
        const reasoningTagHint = isReasoningTagProvider(provider);
        const userTimezone = resolveUserTimezone(
          params.config?.agents?.defaults?.userTimezone,
        );
        const userTime = formatUserTime(new Date(), userTimezone);
        const { defaultAgentId, sessionAgentId } = resolveSessionAgentIds({
          sessionKey: params.sessionKey,
          config: params.config,
        });
        const isDefaultAgent = sessionAgentId === defaultAgentId;
        const systemPrompt = buildAgentSystemPrompt({
          workspaceDir: effectiveWorkspace,
          defaultThinkLevel: params.thinkLevel as ThinkLevel | undefined,
          reasoningLevel: (params.reasoningLevel ?? "off") as ReasoningLevel,
          extraSystemPrompt: params.extraSystemPrompt,
          ownerNumbers: params.ownerNumbers,
          reasoningTagHint,
          heartbeatPrompt: isDefaultAgent
            ? resolveHeartbeatPrompt(
                params.config?.agents?.defaults?.heartbeat?.prompt,
              )
            : undefined,
          skillsPrompt,
          runtimeInfo,
          sandboxInfo,
          toolNames: tools.map((tool) => tool.name),
          toolSummaries: buildToolSummaryMap(tools),
          userTimezone,
          userTime,
          contextFiles,
        });

        const sessionLock = await acquireSessionWriteLock({
          sessionFile: params.sessionFile,
        });

        let aborted = Boolean(params.abortSignal?.aborted);
        let timedOut = false;
        const abortRun = (isTimeout = false) => {
          aborted = true;
          if (isTimeout) timedOut = true;
          runAbortController.abort();
        };

        let isStreaming = false;
        const queue: string[] = [params.prompt];
        let includeImages = true;
        const queueHandle = {
          queueMessage: async (text: string) => {
            queue.push(text);
          },
          isStreaming: () => isStreaming,
          isCompacting: () => false,
          abort: () => abortRun(),
        };
        registerEmbeddedRun(params.sessionId, queueHandle);

        const onAbort = () => abortRun();
        if (params.abortSignal) {
          if (params.abortSignal.aborted) {
            onAbort();
          } else {
            params.abortSignal.addEventListener("abort", onAbort, {
              once: true,
            });
          }
        }

        const abortTimer = setTimeout(
          () => {
            log.warn(
              `embedded run timeout: runId=${params.runId} sessionId=${params.sessionId} timeoutMs=${params.timeoutMs}`,
            );
            abortRun(true);
          },
          Math.max(1, params.timeoutMs),
        );

        const messagingToolSentTexts: string[] = [];
        const messagingToolSentTextsNormalized: string[] = [];
        const messagingToolSentTargets: MessagingToolSend[] = [];
        const pendingMessagingTexts = new Map<string, string>();
        const pendingMessagingTargets = new Map<string, MessagingToolSend>();
        const toolMetaById = new Map<string, string | undefined>();
        const toolSummaryById = new Set<string>();
        const toolMetas: Array<{ toolName?: string; meta?: string }> = [];

        const trimMessagingSent = () => {
          const maxTexts = 10;
          if (messagingToolSentTexts.length > maxTexts) {
            const overflow = messagingToolSentTexts.length - maxTexts;
            messagingToolSentTexts.splice(0, overflow);
            messagingToolSentTextsNormalized.splice(0, overflow);
          }
          const maxTargets = 10;
          if (messagingToolSentTargets.length > maxTargets) {
            const overflow = messagingToolSentTargets.length - maxTargets;
            messagingToolSentTargets.splice(0, overflow);
          }
        };

        const emitToolSummary = (toolName?: string, meta?: string) => {
          if (!params.onToolResult) return;
          const agg = formatToolAggregate(toolName, meta ? [meta] : undefined);
          const { text: cleanedText, mediaUrls } = parseReplyDirectives(agg);
          if (!cleanedText && (!mediaUrls || mediaUrls.length === 0)) return;
          try {
            void params.onToolResult({
              text: cleanedText,
              mediaUrls: mediaUrls?.length ? mediaUrls : undefined,
            });
          } catch {
            // ignore tool result delivery failures
          }
        };

        const aiTools: ToolSet = {};
        for (const toolDef of tools) {
          const toolName = toolDef.name || "tool";
          aiTools[toolName] = tool({
            description: toolDef.description ?? "",
            inputSchema: jsonSchema(
              toolDef.parameters as Record<string, unknown>,
            ),
            execute: async (input, options) => {
              const toolCallId = options.toolCallId;
              const argsRecord =
                input && typeof input === "object"
                  ? (input as Record<string, unknown>)
                  : {};
              const meta = toolDef.name
                ? inferToolMetaFromArgs(toolDef.name, argsRecord)
                : undefined;
              toolMetaById.set(toolCallId, meta);

              emitAgentEvent({
                runId: params.runId,
                stream: "tool",
                data: {
                  phase: "start",
                  name: toolName,
                  toolCallId,
                  args: argsRecord,
                },
              });
              params.onAgentEvent?.({
                stream: "tool",
                data: { phase: "start", name: toolName, toolCallId },
              });

              const shouldEmitToolEvents =
                params.shouldEmitToolResult?.() ?? true;
              if (
                params.onToolResult &&
                shouldEmitToolEvents &&
                !toolSummaryById.has(toolCallId)
              ) {
                toolSummaryById.add(toolCallId);
                emitToolSummary(toolName, meta);
              }

              if (params.onBlockReplyFlush) {
                try {
                  await params.onBlockReplyFlush();
                } catch {
                  // ignore
                }
              }

              if (isMessagingTool(toolName)) {
                const isMessagingSend = isMessagingToolSendAction(
                  toolName,
                  argsRecord,
                );
                if (isMessagingSend) {
                  const sendTarget = extractMessagingToolSend(
                    toolName,
                    argsRecord,
                  );
                  if (sendTarget) {
                    pendingMessagingTargets.set(toolCallId, sendTarget);
                  }
                  const text =
                    (argsRecord.content as string) ??
                    (argsRecord.message as string);
                  if (text && typeof text === "string") {
                    pendingMessagingTexts.set(toolCallId, text);
                  }
                }
              }

              let result: AgentToolResult<unknown>;
              try {
                result = await toolDef.execute(
                  toolCallId,
                  input as never,
                  options.abortSignal,
                );
              } catch (err) {
                const message = describeUnknownError(err);
                emitAgentEvent({
                  runId: params.runId,
                  stream: "tool",
                  data: {
                    phase: "result",
                    name: toolName,
                    toolCallId,
                    meta,
                    isError: true,
                    result: { error: message },
                  },
                });
                params.onAgentEvent?.({
                  stream: "tool",
                  data: {
                    phase: "result",
                    name: toolName,
                    toolCallId,
                    isError: true,
                  },
                });
                return { status: "error", tool: toolName, error: message };
              }

              const isError = isToolResultError(result);
              const pendingText = pendingMessagingTexts.get(toolCallId);
              const pendingTarget = pendingMessagingTargets.get(toolCallId);
              if (pendingText) {
                pendingMessagingTexts.delete(toolCallId);
                if (!isError) {
                  messagingToolSentTexts.push(pendingText);
                  messagingToolSentTextsNormalized.push(
                    normalizeTextForComparison(pendingText),
                  );
                  trimMessagingSent();
                }
              }
              if (pendingTarget) {
                pendingMessagingTargets.delete(toolCallId);
                if (!isError) {
                  messagingToolSentTargets.push(pendingTarget);
                  trimMessagingSent();
                }
              }

              toolMetas.push({ toolName, meta });
              toolMetaById.delete(toolCallId);
              toolSummaryById.delete(toolCallId);

              emitAgentEvent({
                runId: params.runId,
                stream: "tool",
                data: {
                  phase: "result",
                  name: toolName,
                  toolCallId,
                  meta,
                  isError,
                  result: buildToolOutput(result),
                },
              });
              params.onAgentEvent?.({
                stream: "tool",
                data: {
                  phase: "result",
                  name: toolName,
                  toolCallId,
                  meta,
                  isError,
                },
              });

              return buildToolOutput(result);
            },
          });
        }

        let finalAssistantMessage: AssistantMessage | undefined;
        let assistantTexts: string[] = [];
        let lastUsage: Usage | undefined;
        let finishReason: string | undefined;
        let runError: unknown = null;

        try {
          while (queue.length > 0 && !aborted) {
            const promptText = queue.shift() ?? "";
            if (!promptText && (!params.images || params.images.length === 0)) {
              continue;
            }

            await ensureSessionHeader({
              sessionFile: params.sessionFile,
              sessionId: params.sessionId,
              cwd: effectiveWorkspace,
            });

            const sessionManager = guardSessionManager(
              SessionManager.open(params.sessionFile),
            );

            const userContentBlocks: Array<TextContent | ImageContent> = [];
            if (promptText) {
              userContentBlocks.push({ type: "text", text: promptText });
            }
            if (includeImages) {
              for (const image of params.images ?? []) {
                userContentBlocks.push(image);
              }
              includeImages = false;
            }
            const userContent =
              userContentBlocks.length === 1 &&
              userContentBlocks[0]?.type === "text"
                ? promptText
                : userContentBlocks;
            const userMessage: UserMessage = {
              role: "user",
              content: userContent,
              timestamp: Date.now(),
            };
            sessionManager.appendMessage(userMessage);

            let messages = sessionManager.buildSessionContext().messages;
            messages = await sanitizeSessionMessagesImages(
              messages,
              "session:history",
            );
            messages = sanitizeToolUseResultPairing(messages);
            const limited = limitHistoryTurns(
              messages,
              getDmHistoryLimitFromSessionKey(params.sessionKey, params.config),
            );

            const aiMessages = convertAgentMessagesToAi(limited);

            const blockReplyBreak = params.blockReplyBreak ?? "text_end";
            const blockChunking = params.blockReplyChunking;
            const blockChunker = blockChunking
              ? new EmbeddedBlockChunker(blockChunking)
              : null;
            const blockState: BlockTagState = { thinking: false, final: false };
            let blockBuffer = "";
            let deltaBuffer = "";
            let lastStreamedAssistant = "";
            const includeReasoning = params.reasoningLevel === "on";
            const shouldEmitPartialReplies = !(
              includeReasoning && !params.onBlockReply
            );
            const streamReasoning =
              includeReasoning && params.onReasoningStream;
            const emitReasoningStream = (text: string) => {
              const formatted = formatReasoningMessage(text);
              if (!formatted) return;
              try {
                void params.onReasoningStream?.({ text: formatted });
              } catch {
                // ignore
              }
            };

            const emitBlockChunk = (text: string) => {
              if (!params.onBlockReply) return;
              const chunk = stripBlockTags(
                text,
                blockState,
                params.enforceFinalTag,
              ).trimEnd();
              if (!chunk) return;
              const normalizedChunk = normalizeTextForComparison(chunk);
              if (
                isMessagingToolDuplicateNormalized(
                  normalizedChunk,
                  messagingToolSentTextsNormalized,
                )
              ) {
                return;
              }
              const {
                text: cleanedText,
                mediaUrls,
                audioAsVoice,
              } = parseReplyDirectives(chunk);
              if (!cleanedText && (!mediaUrls || mediaUrls.length === 0))
                return;
              try {
                void params.onBlockReply({
                  text: cleanedText,
                  mediaUrls: mediaUrls?.length ? mediaUrls : undefined,
                  audioAsVoice,
                });
              } catch {
                // ignore
              }
            };

            const flushBlockBuffer = (force: boolean) => {
              if (blockChunker) {
                blockChunker.drain({ force, emit: emitBlockChunk });
                if (force) blockChunker.reset();
                return;
              }
              if (!force && blockBuffer.length === 0) return;
              if (blockBuffer.length > 0) {
                emitBlockChunk(blockBuffer);
                blockBuffer = "";
              }
            };

            isStreaming = true;
            const result = streamText({
              model: languageModel,
              system: systemPrompt,
              messages: aiMessages,
              tools: aiTools,
              temperature: extraParams.temperature,
              maxOutputTokens: extraParams.maxOutputTokens,
              abortSignal: runAbortController.signal,
            });

            for await (const event of result.fullStream) {
              if (event.type === "text-delta") {
                deltaBuffer += event.text;
                if (blockChunker) {
                  blockChunker.append(event.text);
                } else {
                  blockBuffer += event.text;
                }

                if (streamReasoning) {
                  emitReasoningStream(
                    extractThinkingFromTaggedStream(deltaBuffer),
                  );
                }

                const next = stripBlockTags(
                  deltaBuffer,
                  { thinking: false, final: false },
                  params.enforceFinalTag,
                ).trim();
                if (next && next !== lastStreamedAssistant) {
                  lastStreamedAssistant = next;
                  const { text: cleanedText, mediaUrls } =
                    parseReplyDirectives(next);
                  emitAgentEvent({
                    runId: params.runId,
                    stream: "assistant",
                    data: {
                      text: cleanedText,
                      mediaUrls: mediaUrls?.length ? mediaUrls : undefined,
                    },
                  });
                  params.onAgentEvent?.({
                    stream: "assistant",
                    data: {
                      text: cleanedText,
                      mediaUrls: mediaUrls?.length ? mediaUrls : undefined,
                    },
                  });
                  if (params.onPartialReply && shouldEmitPartialReplies) {
                    void params.onPartialReply({
                      text: cleanedText,
                      mediaUrls: mediaUrls?.length ? mediaUrls : undefined,
                    });
                  }
                }

                if (blockChunker && blockReplyBreak === "text_end") {
                  blockChunker.drain({ force: false, emit: emitBlockChunk });
                }
              }

              if (event.type === "reasoning-delta" && streamReasoning) {
                emitReasoningStream(event.text);
              }

              if (event.type === "text-end" && blockReplyBreak === "text_end") {
                flushBlockBuffer(true);
              }

              if (
                event.type === "tool-input-start" ||
                event.type === "tool-call"
              ) {
                if (params.onBlockReplyFlush) {
                  try {
                    await params.onBlockReplyFlush();
                  } catch {
                    // ignore
                  }
                }
                flushBlockBuffer(true);
              }

              if (event.type === "finish") {
                finishReason = event.finishReason;
              }
            }

            const response = await result.response;
            const totalUsage = await result.totalUsage;
            finishReason = finishReason ?? (await result.finishReason);
            const normalizedUsage = normalizeUsageFromAi({
              inputTokens: totalUsage.inputTokens,
              outputTokens: totalUsage.outputTokens,
              totalTokens: totalUsage.totalTokens,
              cache_read_input_tokens:
                totalUsage.inputTokenDetails.cacheReadTokens,
              cache_creation_input_tokens:
                totalUsage.inputTokenDetails.cacheWriteTokens,
            });
            lastUsage = mapUsageToPi(normalizedUsage);

            for (const message of response.messages) {
              if (message.role === "assistant") {
                finalAssistantMessage = convertAssistantToPi({
                  message,
                  model,
                  usage: lastUsage,
                  stopReason: resolveFinishReason(finishReason),
                });
                sessionManager.appendMessage(finalAssistantMessage as Message);
                continue;
              }
              if (message.role === "tool") {
                const toolMessage = message as ToolModelMessage;
                for (const part of toolMessage.content) {
                  if (part.type !== "tool-result") continue;
                  const toolResult = convertToolResultToPi(
                    part as ToolResultPart,
                  );
                  sessionManager.appendMessage(toolResult as Message);
                }
              }
            }

            const finalText = finalAssistantMessage
              ? stripBlockTags(
                  extractAssistantText(finalAssistantMessage),
                  { thinking: false, final: false },
                  params.enforceFinalTag,
                ).trim()
              : "";
            if (finalText) assistantTexts = [finalText];

            if (blockReplyBreak === "message_end") {
              flushBlockBuffer(true);
            }

            if (aborted) break;
          }
        } catch (err) {
          if (isAbortError(err)) {
            aborted = true;
          } else {
            runError = err;
          }
        } finally {
          isStreaming = false;
          clearTimeout(abortTimer);
          unregisterEmbeddedRun(params.sessionId, queueHandle);
          await sessionLock.release();
          params.abortSignal?.removeEventListener?.("abort", onAbort);
        }

        if (runError) {
          const message = describeUnknownError(runError);
          const errorText = isContextOverflowError(message)
            ? "Context overflow: the conversation history is too large for the model. " +
              "Use /new or /reset to start a fresh session, or try a model with a larger context window."
            : message;
          return {
            payloads: [{ text: errorText, isError: true }],
            meta: {
              durationMs: Date.now() - started,
              agentMeta: {
                sessionId: params.sessionId,
                provider,
                model: model.id,
              },
            },
            didSendViaMessagingTool: messagingToolSentTexts.length > 0,
            messagingToolSentTexts,
            messagingToolSentTargets,
          };
        }

        if (!finalAssistantMessage) {
          if (aborted || timedOut) {
            return {
              meta: {
                durationMs: Date.now() - started,
                agentMeta: {
                  sessionId: params.sessionId,
                  provider,
                  model: model.id,
                },
                aborted: true,
              },
              didSendViaMessagingTool: messagingToolSentTexts.length > 0,
              messagingToolSentTexts,
              messagingToolSentTargets,
            };
          }
          return {
            payloads: [
              {
                text: "No response generated.",
                isError: true,
              },
            ],
            meta: {
              durationMs: Date.now() - started,
              agentMeta: {
                sessionId: params.sessionId,
                provider,
                model: model.id,
              },
            },
          };
        }

        const usage = normalizeUsageFromAi({
          input: lastUsage?.input,
          output: lastUsage?.output,
          cacheRead: lastUsage?.cacheRead,
          cacheWrite: lastUsage?.cacheWrite,
          total: lastUsage?.totalTokens,
        } as UsageLike);
        const agentMeta: EmbeddedPiAgentMeta = {
          sessionId: params.sessionId,
          provider,
          model: model.id,
          usage,
        };

        const replyItems: Array<{
          text: string;
          media?: string[];
          isError?: boolean;
          audioAsVoice?: boolean;
          replyToId?: string;
          replyToTag?: boolean;
          replyToCurrent?: boolean;
        }> = [];

        const inlineToolResults =
          params.verboseLevel === "on" &&
          !params.onPartialReply &&
          !params.onToolResult &&
          toolMetas.length > 0;
        if (inlineToolResults) {
          for (const { toolName, meta } of toolMetas) {
            const agg = formatToolAggregate(toolName, meta ? [meta] : []);
            const {
              text: cleanedText,
              mediaUrls,
              audioAsVoice,
              replyToId,
              replyToTag,
              replyToCurrent,
            } = parseReplyDirectives(agg);
            if (cleanedText)
              replyItems.push({
                text: cleanedText,
                media: mediaUrls,
                audioAsVoice,
                replyToId,
                replyToTag,
                replyToCurrent,
              });
          }
        }

        const reasoningText =
          params.reasoningLevel === "on"
            ? formatReasoningMessage(
                extractAssistantThinking(finalAssistantMessage) ||
                  extractThinkingFromTaggedStream(
                    extractAssistantText(finalAssistantMessage),
                  ),
              )
            : "";
        if (reasoningText) replyItems.push({ text: reasoningText });

        const fallbackAnswerText = assistantTexts.length
          ? ""
          : stripBlockTags(
              extractAssistantText(finalAssistantMessage),
              { thinking: false, final: false },
              params.enforceFinalTag,
            ).trim();
        const answerTexts = assistantTexts.length
          ? assistantTexts
          : fallbackAnswerText
            ? [fallbackAnswerText]
            : [];
        for (const text of answerTexts) {
          const {
            text: cleanedText,
            mediaUrls,
            audioAsVoice,
            replyToId,
            replyToTag,
            replyToCurrent,
          } = parseReplyDirectives(text);
          if (
            !cleanedText &&
            (!mediaUrls || mediaUrls.length === 0) &&
            !audioAsVoice
          )
            continue;
          replyItems.push({
            text: cleanedText,
            media: mediaUrls,
            audioAsVoice,
            replyToId,
            replyToTag,
            replyToCurrent,
          });
        }

        const hasAudioAsVoiceTag = replyItems.some((item) => item.audioAsVoice);
        const payloads = replyItems
          .map((item) => ({
            text: item.text?.trim() ? item.text.trim() : undefined,
            mediaUrls: item.media?.length ? item.media : undefined,
            mediaUrl: item.media?.[0],
            isError: item.isError,
            replyToId: item.replyToId,
            replyToTag: item.replyToTag,
            replyToCurrent: item.replyToCurrent,
            audioAsVoice:
              item.audioAsVoice || (hasAudioAsVoiceTag && item.media?.length),
          }))
          .filter(
            (p) =>
              p.text || p.mediaUrl || (p.mediaUrls && p.mediaUrls.length > 0),
          );

        const meta = {
          durationMs: Date.now() - started,
          agentMeta,
          aborted: aborted || timedOut ? true : undefined,
        };

        return {
          payloads: payloads.length ? payloads : undefined,
          meta,
          didSendViaMessagingTool: messagingToolSentTexts.length > 0,
          messagingToolSentTexts,
          messagingToolSentTargets,
        };
      } finally {
        restoreSkillEnv?.();
        process.chdir(prevCwd);
      }
    }),
  );
}
