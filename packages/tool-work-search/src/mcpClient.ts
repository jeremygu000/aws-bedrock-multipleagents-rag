export interface WorkSearchResult {
  winfkey: string;
  title: string;
  writers: string[];
  iswc?: string;
  confidence?: number;
}

export interface WorkSearchInput {
  title?: string;
  writer?: string;
  iswc?: string;
  isrc?: string;
  publishers?: string[];
  topK?: number;
}

export const callWorkSearchMcp = async (
  input: WorkSearchInput,
): Promise<{ results: WorkSearchResult[] }> => ({
  results: [
    {
      winfkey: "WINF123456",
      title: input.title ?? "UNKNOWN",
      writers: input.writer ? [input.writer] : [],
      iswc: input.iswc,
      confidence: 0.42,
    },
  ],
});
