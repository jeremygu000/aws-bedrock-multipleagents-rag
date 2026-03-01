import type { Tracer } from "@aws-lambda-powertools/tracer";
import {
  BedrockAgentRuntimeClient,
  RerankCommand,
  type RerankCommandInput,
} from "@aws-sdk/client-bedrock-agent-runtime";
import {
  captureAsync,
  type RerankItem,
  type RerankResultItem,
} from "@aws-bedrock-multiagents/shared";

const RERANK_MODEL_ARN = `arn:aws:bedrock:${process.env.AWS_REGION ?? "us-east-1"}::foundation-model/amazon.rerank-v1:0`;

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
