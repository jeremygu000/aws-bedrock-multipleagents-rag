import {
  captureAsync,
  createObservability,
  createJsonResponse,
  getJsonProps,
  type BedrockActionEvent,
  type BedrockActionResponse,
  type RerankItem,
} from "@aws-bedrock-multiagents/shared";
import { MetricUnit } from "@aws-lambda-powertools/metrics";

import { detectIntent } from "./intentDetector";
import { rerankResults } from "./reranker";

interface DetectIntentBody {
  inputText?: string;
  sessionId?: string;
}

interface RerankResultsBody {
  query?: string;
  results?: RerankItem[];
  topK?: number;
}

interface LambdaContext {
  awsRequestId: string;
}

const { tracer, logger, metrics } = createObservability({
  serviceName: "supervisor-tool",
  namespace: "SupervisorAgent",
});

export const handler = async (
  event: BedrockActionEvent,
  context: LambdaContext,
): Promise<BedrockActionResponse> => {
  const apiPath = event.apiPath;

  if (apiPath === "/detect_intent") {
    return captureAsync(tracer, "handler.detect_intent", async () => {
      tracer.annotateColdStart();
      tracer.addServiceNameAnnotation();

      const props = getJsonProps<DetectIntentBody>(event);
      const inputText = String(props.inputText ?? "");
      const sessionId = String(props.sessionId ?? "unknown");

      logger.appendKeys({
        sessionId,
        agentName: "supervisor-agent",
        actionGroup: event.actionGroup,
        apiPath: event.apiPath,
        requestId: context.awsRequestId,
      });

      metrics.addMetric("IntentDetections", MetricUnit.Count, 1);
      logger.info("detect_intent invoked", { inputText, sessionId });

      const result = await detectIntent(tracer, inputText);

      logger.info("detect_intent result", { result });
      metrics.addMetric(`Intent_${result.intent}`, MetricUnit.Count, 1);
      metrics.publishStoredMetrics();

      return createJsonResponse(event, result, 200);
    });
  }

  if (apiPath === "/rerank_results") {
    return captureAsync(tracer, "handler.rerank_results", async () => {
      tracer.annotateColdStart();
      tracer.addServiceNameAnnotation();

      const props = getJsonProps<RerankResultsBody>(event);
      const query = String(props.query ?? "");
      const items = props.results ?? [];
      const topK = Number(props.topK ?? items.length);

      logger.appendKeys({
        agentName: "supervisor-agent",
        actionGroup: event.actionGroup,
        apiPath: event.apiPath,
        requestId: context.awsRequestId,
      });

      metrics.addMetric("RerankInvocations", MetricUnit.Count, 1);
      logger.info("rerank_results invoked", { query, itemCount: items.length, topK });

      const ranked = await rerankResults(tracer, { query, items, topK });

      logger.info("rerank_results completed", { resultCount: ranked.length });
      metrics.addMetric("RerankResultCount", MetricUnit.Count, ranked.length);
      metrics.publishStoredMetrics();

      return createJsonResponse(event, { results: ranked }, 200);
    });
  }

  logger.warn("Unknown apiPath", { apiPath });
  return createJsonResponse(event, { error: `Unknown path: ${apiPath}` }, 400);
};
