import { RetrievalResult } from "./mergeRerank";

export const bm25Search = async (query: string): Promise<RetrievalResult[]> => [
  {
    id: "bm25-001",
    title: `BM25 result for "${query}"`,
    snippet: "Replace this stub with an OpenSearch BM25 query.",
    score: 0.82,
    source: "bm25",
  },
];
