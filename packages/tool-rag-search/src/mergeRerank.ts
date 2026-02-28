export interface RetrievalResult {
  id: string;
  title: string;
  snippet: string;
  score: number;
  source: "bm25" | "vector";
}

export const mergeAndRerank = (
  ...resultSets: RetrievalResult[][]
): RetrievalResult[] => {
  const deduped = new Map<string, RetrievalResult>();

  for (const result of resultSets.flat()) {
    const existing = deduped.get(result.id);

    if (!existing || result.score > existing.score) {
      deduped.set(result.id, result);
    }
  }

  return [...deduped.values()].sort((left, right) => right.score - left.score);
};
