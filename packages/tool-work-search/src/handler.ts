import {
  BedrockActionEvent,
  BedrockActionResponse,
  createJsonResponse,
  logInfo,
} from "@aws-bedrock-multiagents/shared";

import { searchWork } from "./mcpClient";

export const handler = async (
  event: BedrockActionEvent,
): Promise<BedrockActionResponse> => {
  logInfo("work_search invoked", {
    actionGroup: event.actionGroup,
    apiPath: event.apiPath,
  });

  const results = await searchWork(event.parameters);

  return createJsonResponse(event, {
    results,
  });
};
