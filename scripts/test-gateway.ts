import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { parseArgs } from "node:util";

const HELP_TEXT = `
Usage:
  pnpm test:gateway -- --prompt "Find the work titled Bohemian Rhapsody"

Options:
  --prompt <text>       Prompt to send through the gateway.
  --function-name <n>   Lambda function name. Defaults to GATEWAY_FUNCTION_NAME env.
  --session-id <id>     Session ID. Defaults to a random UUID.
  --no-rerank           Disable reranking step.
  --rerank-top-k <n>    Number of top results after reranking (default: 5).
  --region <region>     AWS region. Defaults to AWS_REGION.
  --help                Show this help message.
`;

const readEnv = (name: string): string | undefined => process.env[name];

const main = async (): Promise<void> => {
  const rawArgs = process.argv.slice(2);
  const args = rawArgs[0] === "--" ? rawArgs.slice(1) : rawArgs;

  const { values, positionals } = parseArgs({
    args,
    allowPositionals: true,
    options: {
      prompt: { type: "string" },
      "function-name": { type: "string" },
      "session-id": { type: "string" },
      "no-rerank": { type: "boolean", default: false },
      "rerank-top-k": { type: "string" },
      region: { type: "string" },
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

  const functionName =
    (values["function-name"] as string | undefined) ?? readEnv("GATEWAY_FUNCTION_NAME");
  if (!functionName) {
    throw new Error(
      "Missing function name. Pass --function-name or set GATEWAY_FUNCTION_NAME env.",
    );
  }

  const region =
    (values.region as string | undefined) ?? readEnv("AWS_REGION") ?? readEnv("AWS_DEFAULT_REGION");

  const client = new LambdaClient({ region });

  const payload = {
    prompt,
    sessionId: values["session-id"] as string | undefined,
    enableRerank: !values["no-rerank"],
    rerankTopK: values["rerank-top-k"] ? Number.parseInt(values["rerank-top-k"] as string, 10) : 5,
  };

  process.stdout.write(`Invoking gateway: ${functionName}\n`);
  process.stdout.write(`Prompt: ${prompt}\n\n`);

  const response = await client.send(
    new InvokeCommand({
      FunctionName: functionName,
      Payload: Buffer.from(JSON.stringify(payload)),
    }),
  );

  if (response.FunctionError) {
    const errorPayload = response.Payload
      ? JSON.parse(new TextDecoder().decode(response.Payload))
      : {};
    process.stderr.write(`Lambda error: ${response.FunctionError}\n`);
    process.stderr.write(`${JSON.stringify(errorPayload, null, 2)}\n`);
    process.exitCode = 1;
    return;
  }

  const result = response.Payload ? JSON.parse(new TextDecoder().decode(response.Payload)) : {};

  process.stdout.write(`sessionId: ${result.sessionId}\n`);
  process.stdout.write(
    `intent: ${result.intent?.type} (confidence: ${result.intent?.confidence})\n`,
  );
  process.stdout.write(`reasoning: ${result.intent?.reasoning}\n\n`);

  if (result.queryRewrite) {
    process.stdout.write(`query rewrite:\n`);
    process.stdout.write(`  original:  ${result.queryRewrite.original}\n`);
    process.stdout.write(`  rewritten: ${result.queryRewrite.rewritten}\n\n`);
  }

  process.stdout.write(`completion:\n${result.completion}\n`);

  if (result.reranked) {
    process.stdout.write(`\nreranked results: ${result.reranked.results?.length ?? 0}\n`);
    for (const item of result.reranked.results ?? []) {
      process.stdout.write(
        `  [${item.originalIndex}] score=${item.relevanceScore.toFixed(4)} ${item.text.slice(0, 100)}\n`,
      );
    }
  }
};

void main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : `Unknown error: ${JSON.stringify(error)}`;
  process.stderr.write(`${message}\n\n${HELP_TEXT}`);
  process.exitCode = 1;
});
