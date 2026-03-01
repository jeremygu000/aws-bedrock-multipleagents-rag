export enum IntentType {
  WORK_SEARCH = "WORK_SEARCH",
  APRA_QA = "APRA_QA",
  AMBIGUOUS = "AMBIGUOUS",
  OUT_OF_SCOPE = "OUT_OF_SCOPE",
}

export interface IntentDetectionResult {
  intent: IntentType;
  confidence: number;
  reasoning: string;
}

export interface RerankItem {
  id: string;
  text: string;
  metadata?: Record<string, unknown>;
}

export interface RerankResultItem {
  id: string;
  text: string;
  relevanceScore: number;
  originalIndex: number;
  metadata?: Record<string, unknown>;
}
