export interface RagSearchInput {
  query: string;
  topK?: number;
  filters?: Record<string, unknown>;
}

export interface RagCitation {
  sourceId: string;
  title?: string;
  url?: string;
  snippet?: string;
}

export interface RagSearchOutput {
  answer: string;
  citations: RagCitation[];
}

export const runEnterpriseRag = async (input: RagSearchInput): Promise<RagSearchOutput> => ({
  answer: `Mock grounded answer for: ${input.query}`,
  citations: [
    {
      sourceId: "doc-001",
      title: "APRA AMCOS FAQ",
      url: "https://example.com/doc-001",
      snippet: "Snippet supporting the answer.",
    },
  ],
});
