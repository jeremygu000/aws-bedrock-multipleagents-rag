import { BedrockActionParameter } from "@aws-bedrock-multiagents/shared";

export interface WorkSearchResult {
  id: string;
  title: string;
  summary: string;
  source: "mcp";
}

const getParameterValue = (
  parameters: BedrockActionParameter[] = [],
  name: string,
): string | undefined => parameters.find((parameter) => parameter.name === name)?.value;

export const searchWork = async (
  parameters: BedrockActionParameter[] = [],
): Promise<WorkSearchResult[]> => {
  const query = getParameterValue(parameters, "query") ?? "empty query";

  // Placeholder for the MCP transport/client integration.
  return [
    {
      id: "work-001",
      title: `Stub work search result for "${query}"`,
      summary: "Replace this with a real MCP call to your work search backend.",
      source: "mcp",
    },
  ];
};
