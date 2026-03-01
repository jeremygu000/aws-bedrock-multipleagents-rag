import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";
import { parseArgs } from "node:util";

import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";

type NamedAgent = "supervisor" | "qa" | "work";

interface ResolvedAgentTarget {
  agent?: NamedAgent;
  agentId: string;
  agentAliasId: string;
  agentAliasArn?: string;
  region: string;
}

interface InvokeResult {
  agentId: string;
  agentAliasId: string;
  agentAliasArn?: string;
  region: string;
  sessionId: string;
  prompt: string;
  completion: string;
  traceCount: number;
  traces: unknown[];
  attributions: unknown[];
  returnControls: unknown[];
  fileEvents: unknown[];
}

const HELP_TEXT = `
Usage:
  pnpm test:agent -- --alias-arn <arn> --prompt "who is APRA AMCOS"

Options:
  --agent <name>             Named agent shortcut: supervisor, qa, or work.
  --alias-arn <arn>          Bedrock agent alias ARN, e.g. arn:aws:bedrock:...:agent-alias/AGENT_ID/ALIAS_ID
  --agent-id <id>            Agent ID. Optional if --alias-arn is provided.
  --alias-id <id>            Agent alias ID. Optional if --alias-arn is provided.
  --prompt <text>            Prompt to send to the agent. If omitted, positional args are joined.
  --session-id <id>          Session ID to reuse across turns. Defaults to a random UUID.
  --region <region>          AWS region. Defaults to BEDROCK_AGENT_REGION, AWS_REGION, or AWS_DEFAULT_REGION.
  --trace                    Enable Bedrock trace collection.
  --stream                   Request streamed final response chunks.
  --guardrail-interval <n>   Optional guardrail interval for streamed responses.
  --output <path>            Save the full JSON result to a file.
  --json                     Print the full JSON result to stdout instead of a human summary.
  --end-session              End the Bedrock agent session after this request.
  --help                     Show this help message.

Environment variable fallbacks:
  BEDROCK_SUPERVISOR_ALIAS_ARN / BEDROCK_SUPERVISOR_AGENT_ID / BEDROCK_SUPERVISOR_ALIAS_ID
  BEDROCK_QA_ALIAS_ARN / BEDROCK_QA_AGENT_ID / BEDROCK_QA_ALIAS_ID
  BEDROCK_WORK_ALIAS_ARN / BEDROCK_WORK_AGENT_ID / BEDROCK_WORK_ALIAS_ID
  BEDROCK_AGENT_ALIAS_ARN
  BEDROCK_AGENT_ID
  BEDROCK_AGENT_ALIAS_ID
  BEDROCK_AGENT_REGION
  BEDROCK_AGENT_SESSION_ID
`;

const NAMED_AGENT_ENV_PREFIX: Record<NamedAgent, string> = {
  supervisor: "BEDROCK_SUPERVISOR",
  qa: "BEDROCK_QA",
  work: "BEDROCK_WORK",
};

const readEnv = (name: string): string | undefined => process.env[name];

const normalizeAgent = (value: string | undefined): NamedAgent | undefined => {
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

const resolveTarget = (
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
    agentId,
    agentAliasId,
    agentAliasArn,
    region,
  };
};

const main = async (): Promise<void> => {
  const rawArgs = process.argv.slice(2);
  const args = rawArgs[0] === "--" ? rawArgs.slice(1) : rawArgs;

  const { values, positionals } = parseArgs({
    args,
    allowPositionals: true,
    options: {
      agent: { type: "string" },
      "alias-arn": { type: "string" },
      "agent-id": { type: "string" },
      "alias-id": { type: "string" },
      prompt: { type: "string" },
      "session-id": { type: "string" },
      region: { type: "string" },
      trace: { type: "boolean", default: false },
      stream: { type: "boolean", default: false },
      "guardrail-interval": { type: "string" },
      output: { type: "string" },
      json: { type: "boolean", default: false },
      "end-session": { type: "boolean", default: false },
      help: { type: "boolean", default: false },
    },
  });

  if (values.help) {
    process.stdout.write(HELP_TEXT);
    return;
  }

  const prompt = (values.prompt as string | undefined) ?? positionals.join(" ").trim();
  if (!prompt) {
    throw new Error("Missing prompt. Pass --prompt or provide positional text.");
  }

  const target = resolveTarget(values);
  const sessionId =
    (values["session-id"] as string | undefined) ??
    readEnv("BEDROCK_AGENT_SESSION_ID") ??
    randomUUID();

  const guardrailIntervalRaw = values["guardrail-interval"] as string | undefined;
  const guardrailInterval = guardrailIntervalRaw
    ? Number.parseInt(guardrailIntervalRaw, 10)
    : undefined;
  if (guardrailIntervalRaw && Number.isNaN(guardrailInterval)) {
    throw new Error("--guardrail-interval must be an integer.");
  }

  const client = new BedrockAgentRuntimeClient({
    region: target.region,
  });

  const command = new InvokeAgentCommand({
    agentId: target.agentId,
    agentAliasId: target.agentAliasId,
    sessionId,
    inputText: prompt,
    enableTrace: values.trace,
    endSession: values["end-session"],
    ...(values.stream || guardrailInterval
      ? {
          streamingConfigurations: {
            streamFinalResponse: values.stream,
            ...(guardrailInterval ? { applyGuardrailInterval: guardrailInterval } : {}),
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

  const result: InvokeResult = {
    agentId: target.agentId,
    agentAliasId: target.agentAliasId,
    agentAliasArn: target.agentAliasArn,
    region: target.region,
    sessionId,
    prompt,
    completion: completionParts.join(""),
    traceCount: traces.length,
    traces,
    attributions,
    returnControls,
    fileEvents,
  };

  const outputPath = values.output as string | undefined;
  if (outputPath) {
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  }

  if (values.json) {
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    return;
  }

  process.stdout.write(`agentId: ${result.agentId}\n`);
  process.stdout.write(`agentAliasId: ${result.agentAliasId}\n`);
  if (target.agent) {
    process.stdout.write(`agent: ${target.agent}\n`);
  }
  process.stdout.write(`region: ${result.region}\n`);
  process.stdout.write(`sessionId: ${result.sessionId}\n\n`);
  process.stdout.write(`${result.completion || "[empty completion]"}\n`);

  if (result.traceCount > 0) {
    process.stdout.write(`\ntrace events: ${result.traceCount}\n`);
  }

  if (result.attributions.length > 0) {
    process.stdout.write(`attributions: ${result.attributions.length}\n`);
  }

  if (outputPath) {
    process.stdout.write(`saved: ${outputPath}\n`);
  }
};

void main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : `Unknown error: ${JSON.stringify(error)}`;
  process.stderr.write(`${message}\n\n${HELP_TEXT}`);
  process.exitCode = 1;
});
