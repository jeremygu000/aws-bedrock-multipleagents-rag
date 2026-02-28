import { Logger } from "@aws-lambda-powertools/logger";
import { Metrics } from "@aws-lambda-powertools/metrics";
import { Tracer } from "@aws-lambda-powertools/tracer";

export interface Observability {
  tracer: Tracer;
  logger: Logger;
  metrics: Metrics;
}

export const captureAsync = async <T>(
  tracer: Tracer,
  name: string,
  fn: () => Promise<T>,
): Promise<T> =>
  Promise.resolve(tracer.provider.captureAsyncFunc(name, async () => fn()) as Promise<T>);

export const createObservability = (options?: {
  serviceName?: string;
  namespace?: string;
}): Observability => {
  const serviceName = options?.serviceName ?? "bedrock-tools";
  const namespace = options?.namespace ?? "BedrockTools";

  return {
    tracer: new Tracer({ serviceName }),
    logger: new Logger({ serviceName }),
    metrics: new Metrics({ namespace, serviceName }),
  };
};
