import { RetrievalResult } from "./mergeRerank";

export const vectorSearch = async (query: string): Promise<RetrievalResult[]> => [
  {
    id: "vector-001",
    title: `Vector result for "${query}"`,
    snippet: "Replace this stub with a pgvector similarity search.",
    score: 0.79,
    source: "vector",
  },
];
