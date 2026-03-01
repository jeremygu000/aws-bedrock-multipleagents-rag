import type { Tracer } from "@aws-lambda-powertools/tracer";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand, PutCommand } from "@aws-sdk/lib-dynamodb";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { captureAsync } from "@aws-bedrock-multiagents/shared";

// eslint-disable-next-line node/no-process-env
const MEMORY_TABLE_NAME = process.env["MEMORY_TABLE_NAME"] ?? "";

const ddbClient = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(ddbClient);

const bedrockClient = new BedrockRuntimeClient({});
const MODEL_ID = "amazon.nova-lite-v1:0";

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
}

export interface SessionMemory {
  sessionId: string;
  summary: string;
  messages: ChatMessage[];
  ttl: number; // 30 days expiration
}

const MAX_RECENT_MESSAGES = 10;
const TTL_SECONDS = 30 * 24 * 60 * 60; // 30 days

/**
 * Retrieve the current memory state for a session.
 */
export const getSessionMemory = async (tracer: Tracer, sessionId: string): Promise<SessionMemory> =>
  captureAsync(tracer, "action.memory.get_session", async () => {
    if (!MEMORY_TABLE_NAME) {
      throw new Error("MEMORY_TABLE_NAME environment variable is not set.");
    }

    try {
      const response = await docClient.send(
        new GetCommand({
          TableName: MEMORY_TABLE_NAME,
          Key: { sessionId },
        }),
      );

      if (response.Item) {
        return response.Item as SessionMemory;
      }
    } catch (error) {
      tracer.addErrorAsMetadata(error as Error);
      console.error("Failed to fetch session memory", error);
    }

    // Return empty memory if not found or on error
    return {
      sessionId,
      summary: "",
      messages: [],
      ttl: Math.floor(Date.now() / 1000) + TTL_SECONDS,
    };
  });

/**
 * Generate a new summary by combining the old summary and evicted messages.
 */
const generateRollingSummary = async (
  tracer: Tracer,
  oldSummary: string,
  evictedMessages: ChatMessage[],
): Promise<string> =>
  captureAsync(tracer, "action.memory.generate_summary", async () => {
    const conversationText = evictedMessages
      .map((m) => `${m.role.toUpperCase()}: ${m.text}`)
      .join("\n");

    const prompt = `You are a helpful AI assistant managing conversation history.
Your task is to take a previous summary and a chunk of new conversational turns, and combine them into a single, concise updated summary.
Preserve key facts, user intents, and past context (e.g., search queries, specific entities mentioned, or policies asked about).

Previous Summary:
${oldSummary || "None"}

New Conversation to summarize:
${conversationText}

Output strictly ONLY the updated summary.`;

    const body = JSON.stringify({
      messages: [{ role: "user", content: [{ text: prompt }] }],
      inferenceConfig: { maxTokens: 300, temperature: 0.1 },
    });

    try {
      const response = await bedrockClient.send(
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

      return result.output?.message?.content?.[0]?.text?.trim() ?? oldSummary;
    } catch (error) {
      tracer.addErrorAsMetadata(error as Error);
      console.error("Failed to generate rolling summary", error);
      return oldSummary; // fallback to old summary if inference fails
    }
  });

/**
 * Append a new user/assistant interaction to the memory.
 * If the history exceeds MAX_RECENT_MESSAGES, it triggers a rolling summary update.
 */
export const appendToMemory = async (
  tracer: Tracer,
  memory: SessionMemory,
  userPrompt: string,
  assistantCompletion: string,
): Promise<void> =>
  captureAsync(tracer, "action.memory.append", async () => {
    if (!MEMORY_TABLE_NAME) {
      return;
    }

    let updatedMessages = [...memory.messages];

    if (userPrompt) {
      updatedMessages.push({ role: "user", text: userPrompt });
    }
    if (assistantCompletion) {
      updatedMessages.push({ role: "assistant", text: assistantCompletion });
    }

    let updatedSummary = memory.summary;

    // Check if we exceed the sliding window limit
    if (updatedMessages.length > MAX_RECENT_MESSAGES) {
      // Calculate how many messages to evict to get back exactly to MAX_RECENT_MESSAGES
      // Alternatively, we can evict half the window to prevent frequent API calls. Let's evict down to 5.
      const targetRetained = Math.floor(MAX_RECENT_MESSAGES / 2);
      const evictCount = updatedMessages.length - targetRetained;

      const evictedMessages = updatedMessages.slice(0, evictCount);
      updatedMessages = updatedMessages.slice(evictCount);

      updatedSummary = await generateRollingSummary(tracer, memory.summary, evictedMessages);
    }

    try {
      await docClient.send(
        new PutCommand({
          TableName: MEMORY_TABLE_NAME,
          Item: {
            sessionId: memory.sessionId,
            summary: updatedSummary,
            messages: updatedMessages,
            ttl: Math.floor(Date.now() / 1000) + TTL_SECONDS,
          } satisfies SessionMemory,
        }),
      );
    } catch (error) {
      tracer.addErrorAsMetadata(error as Error);
      console.error("Failed to save session memory", error);
    }
  });
