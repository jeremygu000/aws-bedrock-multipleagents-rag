import type { Tracer } from "@aws-lambda-powertools/tracer";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import {
  captureAsync,
  type IntentDetectionResult,
  IntentType,
} from "@aws-bedrock-multiagents/shared";

const client = new BedrockRuntimeClient({});
const MODEL_ID = "amazon.nova-pro-v1:0";

/**
 * Confidence threshold for AMBIGUOUS classification.
 * If the model returns AMBIGUOUS but confidence >= this value,
 * we override to APRA_QA as the safer default (will trigger RAG search).
 */
const AMBIGUOUS_CONFIDENCE_THRESHOLD = 0.6;

const SYSTEM_PROMPT = `You are an intent classification engine. Analyze the user's input and classify their intent into exactly one of the following categories:

1. WORK_SEARCH: Exploring, finding, or querying specific music works, songs, compositions, writers, composers, publishers, ISWC, ISRC, or work IDs.
2. APRA_QA: Asking questions about APRA AMCOS policies, distribution, royalties, membership, licensing, general music copyright rules, events, programs, grants, people mentioned in APRA content, or any factual question that could be answered from APRA AMCOS knowledge base articles.
3. AMBIGUOUS: The query is genuinely too vague to determine intent (e.g. single words with no context). Prefer APRA_QA over AMBIGUOUS when in doubt.
4. OUT_OF_SCOPE: The query is clearly unrelated to music works, APRA AMCOS, or music copyright/licensing (e.g. general chat, math, recipes).

Examples:
- "What challenge do young artists face?" → APRA_QA (domain knowledge question)
- "Which albums are nominated for Album of the Year?" → APRA_QA (factual question about music)
- "Find a work titled Hello by Adele" → WORK_SEARCH (searching for specific work)
- "What is Zander Hulme's role?" → APRA_QA (factual question about a person)
- "Is Hindley Street Musical Hall wheelchair accessible?" → APRA_QA (factual question about venue)
- "What is the standard notice period for resigning?" → APRA_QA (policy question)
- "How many times has Vanessa Picken attended SXSW?" → APRA_QA (factual question)
- "What is 2+2?" → OUT_OF_SCOPE (unrelated)

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
        const rawIntent = parsed.intent as IntentType;
        const confidence = parsed.confidence ?? 0;

        // Override AMBIGUOUS to APRA_QA when confidence is reasonable —
        // most "ambiguous" queries are actually answerable by the QA agent.
        if (rawIntent === IntentType.AMBIGUOUS && confidence >= AMBIGUOUS_CONFIDENCE_THRESHOLD) {
          return {
            intent: IntentType.APRA_QA,
            confidence,
            reasoning: `Overridden from AMBIGUOUS (confidence ${confidence} >= ${AMBIGUOUS_CONFIDENCE_THRESHOLD}): ${parsed.reasoning}`,
          };
        }

        return {
          intent: rawIntent,
          confidence,
          reasoning: parsed.reasoning,
        };
      }

      throw new Error(`Invalid intent returned by model: ${parsed.intent}`);
    } catch (error) {
      tracer.addErrorAsMetadata(error as Error);
      // Fallback to OUT_OF_SCOPE if model fails or returns invalid JSON
      return {
        intent: IntentType.OUT_OF_SCOPE,
        confidence: 0,
        reasoning: `Classification failed: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  });
