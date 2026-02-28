import {
  captureAsync,
  createObservability,
  createJsonResponse,
  getJsonProps,
  type BedrockActionEvent,
  type BedrockActionResponse,
} from "@aws-bedrock-multiagents/shared";
import { MetricUnit } from "@aws-lambda-powertools/metrics";

import { runEnterpriseRag } from "./ragClient";

interface RagSearchRequestBody {
  query?: string;
  topK?: number;
  filters?: Record<string, unknown>;
  conversationId?: string;
}

interface LambdaContext {
  awsRequestId: string;
}

const { tracer, logger, metrics } = createObservability({
  serviceName: "rag-search-tool",
});

export const handler = async (
  event: BedrockActionEvent,
  context: LambdaContext,
): Promise<BedrockActionResponse> =>
  captureAsync(tracer, "handler.rag_search", async () => {
    tracer.annotateColdStart();
    tracer.addServiceNameAnnotation();

    const props = getJsonProps<RagSearchRequestBody>(event);
    const conversationId = String(props.conversationId ?? "unknown");
    const query = String(props.query ?? "");
    const topK = Number(props.topK ?? 5);

    logger.appendKeys({
      conversationId,
      agentName: "apra-qa-agent",
      actionGroup: event.actionGroup,
      apiPath: event.apiPath,
      requestId: context.awsRequestId,
    });

    metrics.addMetric("RagSearchInvocations", MetricUnit.Count, 1);
    logger.info("rag_search invoked", { query, topK });

    const data = await captureAsync(tracer, "action.rag_search.compose_answer", async () =>
      runEnterpriseRag({
        query,
        topK,
        filters: props.filters,
      }),
    );

    metrics.addMetric("RagSearchSuccess", MetricUnit.Count, 1);
    metrics.publishStoredMetrics();

    return createJsonResponse(event, data, 200);
  });
