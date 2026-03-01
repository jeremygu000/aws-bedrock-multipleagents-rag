import { randomUUID } from "node:crypto";
import {
  captureAsync,
  createObservability,
  type RerankItem,
} from "@aws-bedrock-multiagents/shared";
import { MetricUnit } from "@aws-lambda-powertools/metrics";

import { detectIntent } from "./intentDetector";
import { rewriteQuery } from "./queryRewriter";
import { invokeBedrockAgent, rerankResults } from "./reranker";

// eslint-disable-next-line node/no-process-env -- Lambda env vars injected by CDK
const SUPERVISOR_AGENT_ID = String(process.env["SUPERVISOR_AGENT_ID"] ?? "");
// eslint-disable-next-line node/no-process-env -- Lambda env vars injected by CDK
const SUPERVISOR_ALIAS_ID = String(process.env["SUPERVISOR_ALIAS_ID"] ?? "");

interface GatewayEvent {
  prompt: string;
  sessionId?: string;
  enableRerank?: boolean;
  rerankTopK?: number;
}

interface GatewayResponse {
  sessionId: string;
  intent: {
    type: string;
    confidence: number;
    reasoning: string;
  };
  queryRewrite: {
    original: string;
    rewritten: string;
  };
  completion: string;
  reranked?: {
    results: Array<{
      id: string;
      text: string;
      relevanceScore: number;
      originalIndex: number;
    }>;
  };
}

interface LambdaContext {
  awsRequestId: string;
}

const { tracer, logger, metrics } = createObservability({
  serviceName: "supervisor-gateway",
  namespace: "SupervisorGateway",
});

export const handler = async (
  event: GatewayEvent,
  context: LambdaContext,
): Promise<GatewayResponse> =>
  captureAsync(tracer, "handler.gateway", async () => {
    tracer.annotateColdStart();
    tracer.addServiceNameAnnotation();

    const prompt = event.prompt;
    const sessionId = event.sessionId ?? randomUUID();
    const enableRerank = event.enableRerank ?? true;
    const rerankTopK = event.rerankTopK ?? 5;

    logger.appendKeys({
      sessionId,
      agentName: "supervisor-gateway",
      requestId: context.awsRequestId,
    });

    logger.info("gateway invoked", { prompt, sessionId, enableRerank });
    metrics.addMetric("GatewayInvocations", MetricUnit.Count, 1);

    // Step 1: Intent Detection
    const intentResult = await detectIntent(tracer, prompt);

    logger.info("intent detected", { intentResult });
    metrics.addMetric(`Intent_${intentResult.intent}`, MetricUnit.Count, 1);

    // Step 2: Query Rewrite
    const rewriteResult = await rewriteQuery(tracer, prompt, intentResult.intent);

    logger.info("query rewritten", { rewriteResult });
    metrics.addMetric("QueryRewrites", MetricUnit.Count, 1);

    // Step 3: Invoke Supervisor Agent (with rewritten query)
    const agentResult = await invokeBedrockAgent(tracer, {
      agentId: SUPERVISOR_AGENT_ID,
      agentAliasId: SUPERVISOR_ALIAS_ID,
      sessionId,
      prompt: rewriteResult.rewritten,
    });

    logger.info("supervisor response received", {
      completionLength: agentResult.completion.length,
    });

    // Step 3: Rerank (if enabled and completion has content)
    let reranked: GatewayResponse["reranked"];
    if (enableRerank && agentResult.completion.length > 0) {
      try {
        const items: RerankItem[] = [
          {
            id: "completion-0",
            text: agentResult.completion,
          },
        ];

        const ranked = await rerankResults(tracer, {
          query: prompt,
          items,
          topK: rerankTopK,
        });

        reranked = { results: ranked };
        metrics.addMetric("RerankInvocations", MetricUnit.Count, 1);
      } catch (rerankError) {
        logger.warn("rerank failed, returning unranked results", {
          error: rerankError instanceof Error ? rerankError.message : String(rerankError),
        });
        metrics.addMetric("RerankFailed", MetricUnit.Count, 1);
      }
    }

    metrics.addMetric("GatewaySuccess", MetricUnit.Count, 1);
    metrics.publishStoredMetrics();

    return {
      sessionId,
      intent: {
        type: intentResult.intent,
        confidence: intentResult.confidence,
        reasoning: intentResult.reasoning,
      },
      queryRewrite: {
        original: rewriteResult.original,
        rewritten: rewriteResult.rewritten,
      },
      completion: agentResult.completion,
      reranked,
    };
  });
