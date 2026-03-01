import { randomUUID } from "node:crypto";

import chalk from "chalk";
import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";

export type NamedAgent = "supervisor" | "qa" | "work";

export interface ResolvedAgentTarget {
  agent?: NamedAgent;
  agentId: string;
  agentAliasId: string;
  agentAliasArn?: string;
  region: string;
}

export interface InvokeAgentOptions {
  target: ResolvedAgentTarget;
  prompt: string;
  sessionId?: string;
  enableTrace?: boolean;
  endSession?: boolean;
  streamFinalResponse?: boolean;
  guardrailInterval?: number;
}

export interface InvokeResult {
  agent?: NamedAgent;
  agentId: string;
  agentAliasId: string;
  agentAliasArn?: string;
  region: string;
  sessionId: string;
  prompt: string;
  completion: string;
  traceCount: number;
  traceAggregateSummary: TraceAggregateSummary;
  traceTimeline: TraceTimelineEntry[];
  traceSummary: string[];
  traces: unknown[];
  attributions: unknown[];
  returnControls: unknown[];
  fileEvents: unknown[];
}

export interface TraceAggregateSummary {
  routingConclusion: string;
  routingPath: string[];
  actionGroups: string[];
  actionGroupResults: string[];
  finalAnswer: string;
}

export interface TraceTimelineEntry {
  index: number;
  stepType: string;
  stepLabel: string;
  time?: string;
  relativeMs?: number;
  traceId?: string;
  collaboratorName?: string;
  invokedCollaborators: string[];
  invokedActionGroups: string[];
  actionOutputs: string[];
  collaboratorOutputs: string[];
  details: string[];
}

const NAMED_AGENT_ENV_PREFIX: Record<NamedAgent, string> = {
  supervisor: "BEDROCK_SUPERVISOR",
  qa: "BEDROCK_QA",
  work: "BEDROCK_WORK",
};

const readEnv = (name: string): string | undefined => process.env[name];

const parseAliasArn = (aliasArn: string): { agentId: string; agentAliasId: string } | undefined => {
  const match = aliasArn.match(/:agent-alias\/([^/]+)\/([^/]+)$/);
  if (!match) {
    return undefined;
  }

  return {
    agentId: match[1],
    agentAliasId: match[2],
  };
};

const decodeChunkBytes = (bytes: Uint8Array): string => new TextDecoder("utf-8").decode(bytes);

const toRecord = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : undefined;

const toArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const toStringValue = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

const toNumberValue = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

const toSnippet = (value: string | undefined, maxLength = 120): string | undefined => {
  if (!value) {
    return undefined;
  }

  const normalized = value.replaceAll(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength - 1)}...`;
};

const uniqueStrings = (values: string[]): string[] => [...new Set(values)];

const requireValue = (value: string | undefined, message: string): string => {
  if (!value) {
    throw new Error(message);
  }

  return value;
};

export const normalizeAgent = (value: string | undefined): NamedAgent | undefined => {
  if (!value) {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();

  if (["supervisor", "supervisor-agent"].includes(normalized)) {
    return "supervisor";
  }

  if (["qa", "apra", "apra-qa", "apra-qa-agent"].includes(normalized)) {
    return "qa";
  }

  if (["work", "work-search", "work-search-agent"].includes(normalized)) {
    return "work";
  }

  throw new Error(`Unsupported --agent value: ${value}. Use supervisor, qa, or work.`);
};

const getNamedAgentEnv = (
  agent: NamedAgent | undefined,
): {
  agentAliasArn?: string;
  agentId?: string;
  agentAliasId?: string;
  region?: string;
} => {
  if (!agent) {
    return {};
  }

  const prefix = NAMED_AGENT_ENV_PREFIX[agent];

  return {
    agentAliasArn: readEnv(`${prefix}_ALIAS_ARN`),
    agentId: readEnv(`${prefix}_AGENT_ID`),
    agentAliasId: readEnv(`${prefix}_ALIAS_ID`),
    region: readEnv(`${prefix}_REGION`),
  };
};

export const resolveTarget = (
  values: Record<string, string | boolean | undefined>,
): ResolvedAgentTarget => {
  const agent = normalizeAgent(values.agent as string | undefined);
  const namedAgentEnv = getNamedAgentEnv(agent);

  const agentAliasArn =
    (values["alias-arn"] as string | undefined) ??
    namedAgentEnv.agentAliasArn ??
    readEnv("BEDROCK_AGENT_ALIAS_ARN");
  const aliasArnParts = agentAliasArn ? parseAliasArn(agentAliasArn) : undefined;

  const agentId =
    (values["agent-id"] as string | undefined) ??
    namedAgentEnv.agentId ??
    readEnv("BEDROCK_AGENT_ID") ??
    aliasArnParts?.agentId;
  const agentAliasId =
    (values["alias-id"] as string | undefined) ??
    namedAgentEnv.agentAliasId ??
    readEnv("BEDROCK_AGENT_ALIAS_ID") ??
    aliasArnParts?.agentAliasId;

  const regionFromArn = agentAliasArn?.split(":")[3];
  const region =
    (values.region as string | undefined) ??
    namedAgentEnv.region ??
    readEnv("BEDROCK_AGENT_REGION") ??
    readEnv("AWS_REGION") ??
    readEnv("AWS_DEFAULT_REGION") ??
    regionFromArn;

  const missing: string[] = [];
  if (!agentId) {
    missing.push("--agent-id or --alias-arn");
  }
  if (!agentAliasId) {
    missing.push("--alias-id or --alias-arn");
  }
  if (!region) {
    missing.push("--region or AWS_DEFAULT_REGION");
  }

  if (missing.length > 0) {
    throw new Error(`Missing required settings: ${missing.join(", ")}`);
  }

  return {
    agent,
    agentId: requireValue(agentId, "Missing required settings: --agent-id or --alias-arn"),
    agentAliasId: requireValue(
      agentAliasId,
      "Missing required settings: --alias-id or --alias-arn",
    ),
    agentAliasArn,
    region: requireValue(region, "Missing required settings: --region or AWS_DEFAULT_REGION"),
  };
};

const summarizeInvocationInput = (
  invocationInput: unknown,
  timelineEntry: TraceTimelineEntry,
  lines: string[],
): void => {
  for (const entry of toArray(invocationInput)) {
    const record = toRecord(entry);
    if (!record) {
      continue;
    }

    const invocationType = toStringValue(record.invocationType) ?? "UNKNOWN";
    const collaboratorInput = toRecord(record.agentCollaboratorInvocationInput);
    if (collaboratorInput) {
      const collaboratorName =
        toStringValue(collaboratorInput.agentCollaboratorName) ?? "unknown-collaborator";
      const input = toRecord(collaboratorInput.input);
      const inputType = toStringValue(input?.type);
      timelineEntry.invokedCollaborators.push(collaboratorName);
      lines.push(
        `invoke collaborator ${collaboratorName}${inputType ? ` (${inputType.toLowerCase()})` : ""}`,
      );
      continue;
    }

    const actionGroupInput = toRecord(record.actionGroupInvocationInput);
    if (actionGroupInput) {
      const actionGroupName =
        toStringValue(actionGroupInput.actionGroupName) ?? "unknown-action-group";
      const apiPath = toStringValue(actionGroupInput.apiPath);
      timelineEntry.invokedActionGroups.push(actionGroupName);
      lines.push(`invoke action group ${actionGroupName}${apiPath ? ` ${apiPath}` : ""}`);
      continue;
    }

    const knowledgeBaseInput = toRecord(record.knowledgeBaseLookupInput);
    if (knowledgeBaseInput) {
      const knowledgeBaseId =
        toStringValue(knowledgeBaseInput.knowledgeBaseId) ?? "unknown-knowledge-base";
      lines.push(`knowledge base lookup ${knowledgeBaseId}`);
      continue;
    }

    lines.push(`invocation ${invocationType.toLowerCase()}`);
  }
};

const summarizeModelInvocation = (step: Record<string, unknown>, lines: string[]): void => {
  const input = toRecord(step.modelInvocationInput);
  if (!input) {
    return;
  }

  const invocationType = toStringValue(input.type);
  const foundationModel = toStringValue(input.foundationModel);
  const label = invocationType
    ? invocationType.toLowerCase().replaceAll("_", " ")
    : "model invocation";
  const modelLabel = foundationModel ? ` via ${foundationModel}` : "";
  lines.push(`${label}${modelLabel}`);

  const output = toRecord(step.modelInvocationOutput);
  const metadata = toRecord(output?.metadata);
  const usage = toRecord(metadata?.usage);
  const inputTokens = toNumberValue(usage?.inputTokens);
  const outputTokens = toNumberValue(usage?.outputTokens);
  if (inputTokens !== undefined || outputTokens !== undefined) {
    lines.push(`tokens in=${inputTokens ?? "?"} out=${outputTokens ?? "?"}`);
  }
};

const summarizeObservation = (
  step: Record<string, unknown>,
  timelineEntry: TraceTimelineEntry,
  lines: string[],
): void => {
  const observation = toRecord(step.observation);
  if (!observation) {
    return;
  }

  const actionGroupOutput = toRecord(observation.actionGroupInvocationOutput);
  if (actionGroupOutput) {
    const text = toSnippet(toStringValue(actionGroupOutput.text));
    const metadata = toRecord(actionGroupOutput.metadata);
    const usage = toRecord(metadata?.usage);
    if (text) {
      timelineEntry.actionOutputs.push(text);
    }
    const answerPrefix = text ? `action output ${text}` : "action output";
    const inputTokens = toNumberValue(usage?.inputTokens);
    const outputTokens = toNumberValue(usage?.outputTokens);
    lines.push(answerPrefix);
    if (inputTokens !== undefined || outputTokens !== undefined) {
      lines.push(`action tokens in=${inputTokens ?? "?"} out=${outputTokens ?? "?"}`);
    }
  }

  const agentCollaboratorOutput = toRecord(observation.agentCollaboratorInvocationOutput);
  if (agentCollaboratorOutput) {
    const output = toRecord(agentCollaboratorOutput.output);
    const text = toSnippet(toStringValue(output?.text));
    if (text) {
      timelineEntry.collaboratorOutputs.push(text);
    }
    lines.push(`collaborator output${text ? ` ${text}` : ""}`);
  }

  const knowledgeBaseOutput = toRecord(observation.knowledgeBaseLookupOutput);
  if (knowledgeBaseOutput) {
    const retrieved = toArray(knowledgeBaseOutput.retrievedReferences);
    lines.push(`knowledge base results ${retrieved.length}`);
  }
};

const summarizeRationale = (step: Record<string, unknown>, lines: string[]): void => {
  const rationale = toRecord(step.rationale);
  const text = toStringValue(rationale?.text);
  if (text) {
    lines.push(`rationale ${text.replaceAll(/\s+/g, " ").slice(0, 160)}`);
  }
};

const summarizeGuardrail = (step: Record<string, unknown>, lines: string[]): void => {
  const action = toStringValue(step.action);
  const outputs = toArray(step.outputAssessments);
  const inputs = toArray(step.inputAssessments);
  lines.push(
    `guardrail${action ? ` action=${action.toLowerCase()}` : ""} inputAssessments=${inputs.length} outputAssessments=${outputs.length}`,
  );
};

const summarizeFailure = (step: Record<string, unknown>, lines: string[]): void => {
  const failureReasons = [
    toStringValue(step.failureReason),
    toStringValue(step.errorMessage),
    toStringValue(step.error),
  ].filter((value): value is string => Boolean(value));

  if (failureReasons.length > 0) {
    lines.push(`failure ${failureReasons[0]}`);
  }
};

const extractTraceStep = (
  tracePart: Record<string, unknown>,
): { stepType: string; step: Record<string, unknown> } | undefined => {
  const container = toRecord(tracePart.trace);
  if (!container) {
    return undefined;
  }

  for (const [stepType, value] of Object.entries(container)) {
    const step = toRecord(value);
    if (step) {
      return { stepType, step };
    }
  }

  return undefined;
};

const summarizeSingleTrace = (trace: unknown, index: number): TraceTimelineEntry => {
  const tracePart = toRecord(trace);
  if (!tracePart) {
    return {
      index: index + 1,
      stepType: "traceEvent",
      stepLabel: "trace event",
      invokedCollaborators: [],
      invokedActionGroups: [],
      actionOutputs: [],
      collaboratorOutputs: [],
      details: ["unparseable"],
    };
  }

  const timestamp = toStringValue(tracePart.eventTime);
  const collaboratorName = toStringValue(tracePart.collaboratorName);
  const stepInfo = extractTraceStep(tracePart);
  const traceId =
    toStringValue(
      stepInfo?.step.modelInvocationInput && toRecord(stepInfo.step.modelInvocationInput)?.traceId,
    ) ?? toStringValue(toRecord(tracePart.trace)?.traceId);
  const stepLabel = stepInfo
    ? stepInfo.stepType
        .replace(/Trace$/, "")
        .replaceAll(/([a-z])([A-Z])/g, "$1 $2")
        .toLowerCase()
    : "trace event";
  const timelineEntry: TraceTimelineEntry = {
    index: index + 1,
    stepType: stepInfo?.stepType ?? "traceEvent",
    stepLabel,
    time: timestamp,
    traceId,
    collaboratorName,
    invokedCollaborators: [],
    invokedActionGroups: [],
    actionOutputs: [],
    collaboratorOutputs: [],
    details: [],
  };
  const details = timelineEntry.details;
  if (stepInfo) {
    summarizeModelInvocation(stepInfo.step, details);
    summarizeRationale(stepInfo.step, details);
    summarizeInvocationInput(stepInfo.step.invocationInput, timelineEntry, details);
    summarizeObservation(stepInfo.step, timelineEntry, details);
    if (stepInfo.stepType === "guardrailTrace") {
      summarizeGuardrail(stepInfo.step, details);
    }
    if (stepInfo.stepType === "failureTrace") {
      summarizeFailure(stepInfo.step, details);
    }
  }

  if (details.length === 0) {
    const keys = Object.keys(tracePart).sort();
    details.push(`keys=${keys.join(", ")}`);
  }

  return timelineEntry;
};

const buildTraceTimeline = (traces: unknown[]): TraceTimelineEntry[] => {
  const timeline = traces.map((trace, index) => summarizeSingleTrace(trace, index));
  const firstTimestamp = timeline[0]?.time ? Date.parse(timeline[0].time) : Number.NaN;

  if (!Number.isNaN(firstTimestamp)) {
    for (const entry of timeline) {
      const parsed = entry.time ? Date.parse(entry.time) : Number.NaN;
      if (!Number.isNaN(parsed)) {
        entry.relativeMs = parsed - firstTimestamp;
      }
    }
  }

  return timeline;
};

const buildTraceAggregateSummary = (
  timeline: TraceTimelineEntry[],
  completion: string,
): TraceAggregateSummary => {
  const routingPath = uniqueStrings(timeline.flatMap((entry) => entry.invokedCollaborators));
  const actionGroups = uniqueStrings(timeline.flatMap((entry) => entry.invokedActionGroups));
  const actionGroupResults = uniqueStrings(
    timeline.flatMap((entry) => [...entry.actionOutputs, ...entry.collaboratorOutputs]),
  ).slice(0, 3);

  const routingConclusion =
    routingPath.length === 0
      ? "no collaborator hop"
      : routingPath.length === 1
        ? routingPath[0]
        : routingPath.join(" -> ");

  return {
    routingConclusion,
    routingPath,
    actionGroups,
    actionGroupResults,
    finalAnswer: toSnippet(completion, 180) ?? "[empty completion]",
  };
};

const getStepColor = (stepType: string): ((text: string) => string) => {
  switch (stepType) {
    case "routingClassifierTrace":
      return chalk.cyan;
    case "orchestrationTrace":
      return chalk.blue;
    case "guardrailTrace":
      return chalk.yellow;
    case "failureTrace":
      return chalk.red;
    case "preProcessingTrace":
      return chalk.magenta;
    case "postProcessingTrace":
      return chalk.green;
    default:
      return chalk.white;
  }
};

export const formatTraceSummary = (traces: unknown[]): string[] =>
  buildTraceTimeline(traces).flatMap((entry) => {
    const headerParts = [`${entry.index}. ${entry.stepLabel}`];
    if (entry.relativeMs !== undefined) {
      headerParts.push(`+${entry.relativeMs}ms`);
    }
    if (entry.collaboratorName) {
      headerParts.push(`collaborator=${entry.collaboratorName}`);
    }

    return [headerParts.join(" "), ...entry.details.map((detail) => `   ${detail}`)];
  });

export const renderTraceTimeline = (timeline: TraceTimelineEntry[]): string[] =>
  timeline.flatMap((entry, index) => {
    const color = getStepColor(entry.stepType);
    const connector = index === timeline.length - 1 ? "└─" : "├─";
    const headerParts = [color(`${connector} ${entry.index}. ${entry.stepLabel}`)];
    if (entry.relativeMs !== undefined) {
      headerParts.push(chalk.dim(`+${entry.relativeMs}ms`));
    }
    if (entry.collaboratorName) {
      headerParts.push(chalk.magenta(`collaborator=${entry.collaboratorName}`));
    }
    if (entry.traceId) {
      headerParts.push(chalk.dim(entry.traceId));
    }

    const lines = [headerParts.join(" ")];
    for (const detail of entry.details) {
      lines.push(`│  ${chalk.gray(detail)}`);
    }

    return lines;
  });

export const renderTraceAggregateSummary = (summary: TraceAggregateSummary): string[] => {
  const lines = [chalk.bold("trace overview")];
  lines.push(`• ${chalk.cyan("route")} ${summary.routingConclusion}`);
  lines.push(
    `• ${chalk.blue("action groups")} ${
      summary.actionGroups.length > 0 ? summary.actionGroups.join(", ") : "none"
    }`,
  );
  if (summary.actionGroupResults.length > 0) {
    lines.push(`• ${chalk.yellow("action results")} ${summary.actionGroupResults[0]}`);
    for (const result of summary.actionGroupResults.slice(1)) {
      lines.push(`  ${chalk.gray(result)}`);
    }
  } else {
    lines.push(`• ${chalk.yellow("action results")} none`);
  }
  lines.push(`• ${chalk.green("final answer")} ${summary.finalAnswer}`);
  return lines;
};

export const invokeAgent = async (options: InvokeAgentOptions): Promise<InvokeResult> => {
  const sessionId = options.sessionId ?? randomUUID();

  const client = new BedrockAgentRuntimeClient({
    region: options.target.region,
  });

  const command = new InvokeAgentCommand({
    agentId: options.target.agentId,
    agentAliasId: options.target.agentAliasId,
    sessionId,
    inputText: options.prompt,
    enableTrace: options.enableTrace,
    endSession: options.endSession,
    ...(options.streamFinalResponse || options.guardrailInterval
      ? {
          streamingConfigurations: {
            streamFinalResponse: options.streamFinalResponse,
            ...(options.guardrailInterval
              ? { applyGuardrailInterval: options.guardrailInterval }
              : {}),
          },
        }
      : {}),
  });

  const response = await client.send(command);

  const traces: unknown[] = [];
  const attributions: unknown[] = [];
  const returnControls: unknown[] = [];
  const fileEvents: unknown[] = [];
  const completionParts: string[] = [];

  for await (const event of response.completion ?? []) {
    const record = toRecord(event);
    if (!record) {
      continue;
    }

    const chunk = toRecord(record.chunk);
    const bytes = chunk?.bytes;
    if (bytes instanceof Uint8Array) {
      completionParts.push(decodeChunkBytes(bytes));
    }
    if (chunk?.attribution !== undefined) {
      attributions.push(chunk.attribution);
    }

    if (record.trace !== undefined) {
      traces.push(record.trace);
    }

    if (record.returnControl !== undefined) {
      returnControls.push(record.returnControl);
    }

    if (record.files !== undefined) {
      fileEvents.push(record.files);
    }
  }

  const traceTimeline = buildTraceTimeline(traces);
  const completion = completionParts.join("");
  const traceAggregateSummary = buildTraceAggregateSummary(traceTimeline, completion);

  return {
    agent: options.target.agent,
    agentId: options.target.agentId,
    agentAliasId: options.target.agentAliasId,
    agentAliasArn: options.target.agentAliasArn,
    region: options.target.region,
    sessionId,
    prompt: options.prompt,
    completion,
    traceCount: traces.length,
    traceAggregateSummary,
    traceTimeline,
    traceSummary: formatTraceSummary(traces),
    traces,
    attributions,
    returnControls,
    fileEvents,
  };
};
