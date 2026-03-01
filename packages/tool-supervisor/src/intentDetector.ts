import type { Tracer } from "@aws-lambda-powertools/tracer";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import {
  captureAsync,
  type IntentDetectionResult,
  IntentType,
} from "@aws-bedrock-multiagents/shared";

const client = new BedrockRuntimeClient({});
const MODEL_ID = "amazon.nova-lite-v1:0";

const SYSTEM_PROMPT = `You are an intent classification engine. Analyze the user's input and classify their intent into exactly one of the following categories:

1. WORK_SEARCH: Exploring, finding, or querying specific music works, songs, compositions, writers, composers, publishers, ISWC, ISRC, or work IDs.
2. APRA_QA: Asking questions about APRA AMCOS policies, distribution, royalties, membership, licensing, or general music copyright rules.
3. AMBIGUOUS: The query could equally belong to both WORK_SEARCH and APRA_QA, or it's too vague to decide between them.
4. OUT_OF_SCOPE: The query is unrelated to music works, APRA AMCOS, or music copyright/licensing (e.g. general chat, math, recipes, completely unrelated topics).

Respond ONLY with a valid JSON object matching this schema:
{
  "intent": "WORK_SEARCH" | "APRA_QA" | "AMBIGUOUS" | "OUT_OF_SCOPE",
  "confidence": number, // A float between 0.0 and 1.0 representing your classification confidence
  "reasoning": "string" // A brief 1-sentence explanation of why you chose this intent
}

Do NOT include markdown formatting, backticks, or any other text outside the JSON object.`;

export const detectIntent = async (
  tracer: Tracer,
  inputText: string,
): Promise<IntentDetectionResult> =>
  captureAsync(tracer, "action.detect_intent.classify", async () => {
    const body = JSON.stringify({
      messages: [{ role: "user", content: [{ text: inputText }] }],
      system: [{ text: SYSTEM_PROMPT }],
      inferenceConfig: {
        maxTokens: 256,
        temperature: 0.1, // Low temperature for consistent classification
      },
    });

    try {
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

      const textOutput = result.output?.message?.content?.[0]?.text?.trim() ?? "";

      // Parse the JSON output from the model
      const parsed = JSON.parse(textOutput) as {
        intent: string;
        confidence: number;
        reasoning: string;
      };

      // Validate the intent
      if (Object.values(IntentType).includes(parsed.intent as IntentType)) {
        return {
          intent: parsed.intent as IntentType,
          confidence: parsed.confidence,
          reasoning: parsed.reasoning,
        };
      }

      throw new Error(`Invalid intent returned by model: ${parsed.intent}`);
    } catch (error) {
      // Fallback to OUT_OF_SCOPE if model fails or returns invalid JSON
      return {
        intent: IntentType.OUT_OF_SCOPE,
        confidence: 0,
        reasoning: `Classification failed: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  });
