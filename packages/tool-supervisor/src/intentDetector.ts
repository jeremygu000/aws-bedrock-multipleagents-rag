import type { Tracer } from "@aws-lambda-powertools/tracer";
import {
  captureAsync,
  type IntentDetectionResult,
  IntentType,
} from "@aws-bedrock-multiagents/shared";

interface SignalWeights {
  keywords: string[];
  weight: number;
}

const WORK_SEARCH_SIGNALS: SignalWeights[] = [
  { keywords: ["iswc", "isrc"], weight: 0.9 },
  { keywords: ["winfkey", "work id", "work code"], weight: 0.85 },
  { keywords: ["composer", "songwriter", "lyricist", "writer"], weight: 0.6 },
  { keywords: ["publisher", "publishing"], weight: 0.5 },
  { keywords: ["title", "song", "track", "work", "composition", "piece"], weight: 0.4 },
  { keywords: ["search", "find", "look up", "lookup"], weight: 0.2 },
];

const APRA_QA_SIGNALS: SignalWeights[] = [
  { keywords: ["apra", "amcos", "apra amcos"], weight: 0.9 },
  { keywords: ["licence", "license", "licensing"], weight: 0.7 },
  { keywords: ["royalty", "royalties", "distribution"], weight: 0.7 },
  { keywords: ["membership", "member", "join", "register"], weight: 0.6 },
  { keywords: ["copyright", "rights", "permission"], weight: 0.5 },
  { keywords: ["fee", "cost", "price", "pay", "payment"], weight: 0.4 },
  { keywords: ["how", "what", "when", "where", "can i", "do i"], weight: 0.15 },
];

const computeScore = (text: string, signals: SignalWeights[]): number => {
  const lower = text.toLowerCase();
  let total = 0;
  let maxPossible = 0;

  for (const signal of signals) {
    maxPossible += signal.weight;
    if (signal.keywords.some((kw) => lower.includes(kw))) {
      total += signal.weight;
    }
  }

  return maxPossible > 0 ? total / maxPossible : 0;
};

const AMBIGUITY_THRESHOLD = 0.2;
const MIN_CONFIDENCE = 0.15;

export const detectIntent = async (
  tracer: Tracer,
  inputText: string,
): Promise<IntentDetectionResult> =>
  captureAsync(tracer, "action.detect_intent.classify", async () => {
    const workScore = computeScore(inputText, WORK_SEARCH_SIGNALS);
    const qaScore = computeScore(inputText, APRA_QA_SIGNALS);

    const maxScore = Math.max(workScore, qaScore);

    if (maxScore < MIN_CONFIDENCE) {
      return {
        intent: IntentType.OUT_OF_SCOPE,
        confidence: 1 - maxScore,
        reasoning: `No domain signals detected (workScore=${workScore.toFixed(3)}, qaScore=${qaScore.toFixed(3)})`,
      };
    }

    const diff = Math.abs(workScore - qaScore);

    if (diff < AMBIGUITY_THRESHOLD && workScore >= MIN_CONFIDENCE && qaScore >= MIN_CONFIDENCE) {
      return {
        intent: IntentType.AMBIGUOUS,
        confidence: diff,
        reasoning: `Ambiguous: workScore=${workScore.toFixed(3)}, qaScore=${qaScore.toFixed(3)}, diff=${diff.toFixed(3)}`,
      };
    }

    if (workScore > qaScore) {
      return {
        intent: IntentType.WORK_SEARCH,
        confidence: workScore,
        reasoning: `Work search signals dominant: workScore=${workScore.toFixed(3)} > qaScore=${qaScore.toFixed(3)}`,
      };
    }

    return {
      intent: IntentType.APRA_QA,
      confidence: qaScore,
      reasoning: `QA signals dominant: qaScore=${qaScore.toFixed(3)} > workScore=${workScore.toFixed(3)}`,
    };
  });
