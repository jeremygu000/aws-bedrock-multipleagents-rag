import type { Tracer } from "@aws-lambda-powertools/tracer";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { captureAsync } from "@aws-bedrock-multiagents/shared";

const client = new BedrockRuntimeClient({});
const MODEL_ID = "amazon.nova-lite-v1:0";

const SYSTEM_PROMPT = `You are an AI assistant for APRA AMCOS. The user asked a question that is ambiguous or too vague. Your job is to generate a short, friendly clarifying question (1-2 sentences maximum) to ask the user what they meant.

The user's intent is unclear between two possibilities:
1. WORK_SEARCH: Exploring, finding, or querying specific music works, songs, writers, ISWC, ISRC, etc.
2. APRA_QA: Asking questions about APRA AMCOS policies, royalties, licensing, or membership rules.

Generate a polite follow-up asking them to clarify if they meant a search or a policy question, tailoring it slightly to whatever context they provided.

Do NOT answer the question. Only ask for clarification.
Output ONLY the clarifying question text, nothing else.`;

export const generateClarification = async (tracer: Tracer, inputText: string): Promise<string> =>
  captureAsync(tracer, "action.gateway.generate_clarification", async () => {
    const body = JSON.stringify({
      messages: [{ role: "user", content: [{ text: inputText }] }],
      system: [{ text: SYSTEM_PROMPT }],
      inferenceConfig: {
        maxTokens: 150,
        temperature: 0.3, // Slightly higher temperature for conversational tone
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

      const textOutput = result.output?.message?.content?.[0]?.text?.trim();

      if (textOutput) {
        return textOutput;
      }
      throw new Error("Empty response from model");
    } catch (error) {
      tracer.addErrorAsMetadata(error as Error);
      // Fallback if LLM fails
      return "I'm not quite sure what you mean. Are you trying to search for a specific music work, or do you have a question about APRA AMCOS policies and licensing?";
    }
  });
