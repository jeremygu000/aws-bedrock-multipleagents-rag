import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";
import { parseArgs } from "node:util";

import {
  invokeAgent,
  renderTraceAggregateSummary,
  renderTraceTimeline,
  resolveTarget,
} from "./lib/bedrock-agent-client";

interface EvalInputRow {
  id?: string | number;
  prompt?: string;
  question?: string;
  reference?: string;
  groundTruth?: string;
  ground_truth?: string;
  metadata?: Record<string, unknown>;
}

interface EvalOutputRow {
  id: string;
  agent?: string;
  agentId: string;
  agentAliasId: string;
  agentAliasArn?: string;
  region: string;
  sessionId: string;
  prompt: string;
  question: string;
  answer: string;
  reference?: string;
  ground_truth?: string;
  metadata?: Record<string, unknown>;
  traceCount: number;
  traceSummary: string[];
  traces: unknown[];
  attributions: unknown[];
  returnControls: unknown[];
  fileEvents: unknown[];
}

const HELP_TEXT = `
Usage:
  pnpm eval:agent -- --agent supervisor --input evals/prompts.jsonl --output tmp/evals/supervisor.jsonl

Input formats:
  - JSONL: one object per line
  - JSON: an array of objects

Supported input fields:
  - id
  - prompt
  - question
  - reference
  - groundTruth
  - ground_truth
  - metadata

Options:
  --agent <name>             Named agent shortcut: supervisor, qa, or work.
  --alias-arn <arn>          Bedrock agent alias ARN.
  --agent-id <id>            Agent ID. Optional if --alias-arn is provided.
  --alias-id <id>            Agent alias ID. Optional if --alias-arn is provided.
  --region <region>          AWS region. Defaults to BEDROCK_AGENT_REGION, AWS_REGION, or AWS_DEFAULT_REGION.
  --input <path>             Required. Path to a JSON or JSONL evaluation dataset.
  --output <path>            Required. Path to the output JSON or JSONL file.
  --format <json|jsonl>      Output format. Defaults to jsonl.
  --trace-summary            Include trace summary in terminal progress output.
  --fail-fast                Stop on the first failing row.
  --shared-session           Reuse one session ID for all rows.
  --help                     Show this help message.
`;

const parseInput = async (inputPath: string): Promise<EvalInputRow[]> => {
  const raw = await readFile(inputPath, "utf8");
  const trimmed = raw.trim();

  if (!trimmed) {
    return [];
  }

  if (trimmed.startsWith("[")) {
    const parsed = JSON.parse(trimmed) as unknown;
    if (!Array.isArray(parsed)) {
      throw new Error("JSON input must be an array.");
    }

    return parsed as EvalInputRow[];
  }

  return trimmed
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as EvalInputRow);
};

const getPrompt = (row: EvalInputRow): string => {
  const prompt = row.prompt ?? row.question;
  if (!prompt || prompt.trim().length === 0) {
    throw new Error("Each input row must include prompt or question.");
  }

  return prompt;
};

const getReference = (row: EvalInputRow): string | undefined =>
  row.reference ?? row.groundTruth ?? row.ground_truth;

const serializeResults = (results: EvalOutputRow[], format: string): string => {
  if (format === "json") {
    return `${JSON.stringify(results, null, 2)}\n`;
  }

  return `${results.map((row) => JSON.stringify(row)).join("\n")}\n`;
};

const main = async (): Promise<void> => {
  const rawArgs = process.argv.slice(2);
  const args = rawArgs[0] === "--" ? rawArgs.slice(1) : rawArgs;

  const { values } = parseArgs({
    args,
    options: {
      agent: { type: "string" },
      "alias-arn": { type: "string" },
      "agent-id": { type: "string" },
      "alias-id": { type: "string" },
      region: { type: "string" },
      input: { type: "string" },
      output: { type: "string" },
      format: { type: "string" },
      "trace-summary": { type: "boolean", default: false },
      "fail-fast": { type: "boolean", default: false },
      "shared-session": { type: "boolean", default: false },
      help: { type: "boolean", default: false },
    },
  });

  if (values.help) {
    process.stdout.write(HELP_TEXT);
    return;
  }

  const inputPath = values.input as string | undefined;
  const outputPath = values.output as string | undefined;
  const format = (values.format as string | undefined) ?? "jsonl";

  if (!inputPath) {
    throw new Error("Missing required --input path.");
  }
  if (!outputPath) {
    throw new Error("Missing required --output path.");
  }
  if (!["json", "jsonl"].includes(format)) {
    throw new Error("--format must be either json or jsonl.");
  }

  const target = resolveTarget(values);
  const rows = await parseInput(inputPath);
  const sharedSessionId = values["shared-session"] ? randomUUID() : undefined;
  const results: EvalOutputRow[] = [];

  for (const [index, row] of rows.entries()) {
    const prompt = getPrompt(row);
    const id = String(row.id ?? index + 1);

    try {
      const result = await invokeAgent({
        target,
        prompt,
        sessionId: sharedSessionId,
        enableTrace: true,
      });

      results.push({
        id,
        agent: result.agent,
        agentId: result.agentId,
        agentAliasId: result.agentAliasId,
        agentAliasArn: result.agentAliasArn,
        region: result.region,
        sessionId: result.sessionId,
        prompt,
        question: prompt,
        answer: result.completion,
        reference: getReference(row),
        ground_truth: getReference(row),
        metadata: row.metadata,
        traceCount: result.traceCount,
        traceSummary: result.traceSummary,
        traces: result.traces,
        attributions: result.attributions,
        returnControls: result.returnControls,
        fileEvents: result.fileEvents,
      });

      process.stdout.write(`[${index + 1}/${rows.length}] ok id=${id}\n`);
      if (values["trace-summary"] && result.traceSummary.length > 0) {
        for (const line of renderTraceAggregateSummary(result.traceAggregateSummary)) {
          process.stdout.write(`  ${line}\n`);
        }
        process.stdout.write("  \n");
        for (const line of renderTraceTimeline(result.traceTimeline)) {
          process.stdout.write(`  ${line}\n`);
        }
      }
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : `Unknown error: ${JSON.stringify(error)}`;
      process.stderr.write(`[${index + 1}/${rows.length}] failed id=${id}: ${message}\n`);

      if (values["fail-fast"]) {
        throw error;
      }
    }
  }

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, serializeResults(results, format), "utf8");
  process.stdout.write(`saved: ${outputPath}\n`);
  process.stdout.write(`rows: ${results.length}\n`);
};

void main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : `Unknown error: ${JSON.stringify(error)}`;
  process.stderr.write(`${message}\n\n${HELP_TEXT}`);
  process.exitCode = 1;
});
