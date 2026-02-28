import {
  BedrockActionEvent,
  BedrockActionResponse,
  createJsonResponse,
  logInfo,
} from "@aws-bedrock-multiagents/shared";

import { mergeAndRerank } from "./mergeRerank";
import { bm25Search } from "./opensearch";
import { vectorSearch } from "./pgvector";

const getQuery = (event: BedrockActionEvent): string =>
  event.parameters?.find((parameter) => parameter.name === "query")?.value ?? "empty query";

export const handler = async (
  event: BedrockActionEvent,
): Promise<BedrockActionResponse> => {
  const query = getQuery(event);

  logInfo("rag_search invoked", {
    actionGroup: event.actionGroup,
    apiPath: event.apiPath,
    query,
  });

  const [bm25Results, vectorResults] = await Promise.all([
    bm25Search(query),
    vectorSearch(query),
  ]);

  const results = mergeAndRerank(bm25Results, vectorResults);

  return createJsonResponse(event, {
    query,
    results,
  });
};
