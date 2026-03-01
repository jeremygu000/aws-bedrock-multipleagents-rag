import type { Tracer } from "@aws-lambda-powertools/tracer";
import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
  RerankCommand,
  type RerankCommandInput,
} from "@aws-sdk/client-bedrock-agent-runtime";
import {
  captureAsync,
  type RerankItem,
  type RerankResultItem,
} from "@aws-bedrock-multiagents/shared";

// eslint-disable-next-line node/no-process-env -- Lambda env var injected by CDK
const RERANK_MODEL_ARN = String(process.env["RERANK_MODEL_ARN"] ?? "");

const client = new BedrockAgentRuntimeClient({});

export interface RerankInput {
  query: string;
  items: RerankItem[];
  topK?: number;
}

export const rerankResults = async (
  tracer: Tracer,
  input: RerankInput,
): Promise<RerankResultItem[]> =>
  captureAsync(tracer, "action.rerank_results.bedrock_rerank", async () => {
    const { query, items, topK } = input;

    if (items.length === 0) {
      return [];
    }

    const commandInput: RerankCommandInput = {
      queries: [{ textQuery: { text: query }, type: "TEXT" }],
      sources: items.map((item) => ({
        type: "INLINE",
        inlineDocumentSource: {
          type: "TEXT",
          textDocument: { text: item.text },
        },
      })),
      rerankingConfiguration: {
        type: "BEDROCK_RERANKING_MODEL",
        bedrockRerankingConfiguration: {
          modelConfiguration: {
            modelArn: RERANK_MODEL_ARN,
          },
          numberOfResults: topK ?? items.length,
        },
      },
    };

    const response = await client.send(new RerankCommand(commandInput));

    return (response.results ?? []).map((result) => {
      const originalIndex = result.index ?? 0;
      const originalItem = items[originalIndex];
      return {
        id: originalItem?.id ?? String(originalIndex),
        text: originalItem?.text ?? "",
        relevanceScore: result.relevanceScore ?? 0,
        originalIndex,
        metadata: originalItem?.metadata,
      };
    });
  });

export interface InvokeAgentInput {
  agentId: string;
  agentAliasId: string;
  sessionId: string;
  prompt: string;
}

export interface InvokeAgentOutput {
  completion: string;
  sessionId: string;
}

export const invokeBedrockAgent = async (
  tracer: Tracer,
  input: InvokeAgentInput,
): Promise<InvokeAgentOutput> =>
  captureAsync(tracer, "action.gateway.invoke_supervisor", async () => {
    const command = new InvokeAgentCommand({
      agentId: input.agentId,
      agentAliasId: input.agentAliasId,
      sessionId: input.sessionId,
      inputText: input.prompt,
      enableTrace: true,
    });

    const response = await client.send(command);

    const parts: string[] = [];
    for await (const event of response.completion ?? []) {
      const chunk = event as unknown as Record<string, unknown>;
      const chunkData = chunk.chunk as Record<string, unknown> | undefined;
      const bytes = chunkData?.bytes;
      if (bytes instanceof Uint8Array) {
        parts.push(new TextDecoder("utf-8").decode(bytes));
      }
    }

    return {
      completion: parts.join(""),
      sessionId: input.sessionId,
    };
  });
