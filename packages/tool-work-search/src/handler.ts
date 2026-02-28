import {
  captureAsync,
  createObservability,
  createJsonResponse,
  getJsonProps,
  type BedrockActionEvent,
  type BedrockActionResponse,
} from "@aws-bedrock-multiagents/shared";
import { MetricUnit } from "@aws-lambda-powertools/metrics";

import { callWorkSearchMcp } from "./mcpClient";

interface WorkSearchRequestBody {
  title?: string;
  writer?: string;
  iswc?: string;
  isrc?: string;
  publishers?: string[];
  topK?: number;
  conversationId?: string;
}

interface LambdaContext {
  awsRequestId: string;
}

const { tracer, logger, metrics } = createObservability({
  serviceName: "work-search-tool",
});

export const handler = async (
  event: BedrockActionEvent,
  context: LambdaContext,
): Promise<BedrockActionResponse> =>
  captureAsync(tracer, "handler.work_search", async () => {
    tracer.annotateColdStart();
    tracer.addServiceNameAnnotation();

    const props = getJsonProps<WorkSearchRequestBody>(event);
    const conversationId = String(props.conversationId ?? "unknown");

    logger.appendKeys({
      conversationId,
      agentName: "work-search-agent",
      actionGroup: event.actionGroup,
      apiPath: event.apiPath,
      requestId: context.awsRequestId,
    });

    metrics.addMetric("WorkSearchInvocations", MetricUnit.Count, 1);
    logger.info("work_search invoked", { props });

    const data = await captureAsync(tracer, "action.work_search.mcp_call", async () =>
      callWorkSearchMcp({
        title: props.title,
        writer: props.writer,
        iswc: props.iswc,
        isrc: props.isrc,
        publishers: props.publishers,
        topK: props.topK,
      }),
    );

    metrics.addMetric("WorkSearchSuccess", MetricUnit.Count, 1);
    metrics.publishStoredMetrics();

    return createJsonResponse(event, data, 200);
  });
