import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { parseArgs } from "node:util";

import {
  invokeAgent,
  renderTraceAggregateSummary,
  renderTraceTimeline,
  resolveTarget,
} from "./lib/bedrock-agent-client";

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
  --trace                    Enable raw Bedrock trace collection.
  --trace-summary            Print a condensed trace summary in the terminal and include it in JSON output.
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
      "trace-summary": { type: "boolean", default: false },
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

  const guardrailIntervalRaw = values["guardrail-interval"] as string | undefined;
  const guardrailInterval = guardrailIntervalRaw
    ? Number.parseInt(guardrailIntervalRaw, 10)
    : undefined;
  if (guardrailIntervalRaw && Number.isNaN(guardrailInterval)) {
    throw new Error("--guardrail-interval must be an integer.");
  }

  const traceSummaryEnabled = values["trace-summary"];
  const result = await invokeAgent({
    target: resolveTarget(values),
    prompt,
    sessionId: values["session-id"] as string | undefined,
    enableTrace: values.trace || traceSummaryEnabled,
    endSession: values["end-session"],
    streamFinalResponse: values.stream,
    guardrailInterval,
  });

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
  if (result.agent) {
    process.stdout.write(`agent: ${result.agent}\n`);
  }
  process.stdout.write(`region: ${result.region}\n`);
  process.stdout.write(`sessionId: ${result.sessionId}\n\n`);
  process.stdout.write(`${result.completion || "[empty completion]"}\n`);

  if (traceSummaryEnabled && result.traceSummary.length > 0) {
    process.stdout.write("\ntrace summary:\n");
    for (const line of renderTraceAggregateSummary(result.traceAggregateSummary)) {
      process.stdout.write(`${line}\n`);
    }
    process.stdout.write("\n");
    for (const line of renderTraceTimeline(result.traceTimeline)) {
      process.stdout.write(`${line}\n`);
    }
  } else if (result.traceCount > 0) {
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
