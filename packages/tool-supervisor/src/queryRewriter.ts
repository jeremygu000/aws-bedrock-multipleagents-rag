import type { Tracer } from "@aws-lambda-powertools/tracer";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { captureAsync, type IntentType } from "@aws-bedrock-multiagents/shared";

const client = new BedrockRuntimeClient({});
const MODEL_ID = "amazon.nova-lite-v1:0";

const SYSTEM_PROMPTS: Record<string, string> = {
  WORK_SEARCH: [
    "You are a query rewriter for a music work search system.",
    "Rewrite the user query to be more effective for searching a music works database.",
    "Extract and emphasise: title, writer/composer names, ISWC, ISRC, publisher.",
    "Output ONLY the rewritten query, nothing else. No explanation.",
  ].join(" "),
  APRA_QA: [
    "You are a query rewriter for an APRA AMCOS knowledge base.",
    "Rewrite the user query to be a clear, specific question about APRA AMCOS policies, licensing, royalties, or membership.",
    "Output ONLY the rewritten query, nothing else. No explanation.",
  ].join(" "),
};

const DEFAULT_SYSTEM_PROMPT = [
  "You are a query rewriter.",
  "Rewrite the user query to be clearer and more specific.",
  "Output ONLY the rewritten query, nothing else.",
].join(" ");

export interface QueryRewriteResult {
  original: string;
  rewritten: string;
  intent: string;
}

export const rewriteQuery = async (
  tracer: Tracer,
  query: string,
  intent: IntentType,
): Promise<QueryRewriteResult> =>
  captureAsync(tracer, "action.gateway.rewrite_query", async () => {
    const systemPrompt = SYSTEM_PROMPTS[intent] ?? DEFAULT_SYSTEM_PROMPT;

    const body = JSON.stringify({
      messages: [{ role: "user", content: [{ text: query }] }],
      system: [{ text: systemPrompt }],
      inferenceConfig: {
        maxTokens: 256,
        temperature: 0.2,
      },
    });

    const response = await client.send(
      new InvokeModelCommand({
        modelId: MODEL_ID,
        contentType: "application/json",
        accept: "application/json",
        body: Buffer.from(body),
      }),
    );

    const result = JSON.parse(new TextDecoder().decode(response.body)) as {
      output?: { message?: { content?: Array<{ text?: string }> } };
    };

    const rewritten = result.output?.message?.content?.[0]?.text?.trim() ?? query;

    return { original: query, rewritten, intent };
  });
