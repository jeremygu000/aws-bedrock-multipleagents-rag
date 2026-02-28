export interface BedrockActionEvent {
  actionGroup: string;
  apiPath: string;
  httpMethod: string;
  inputText?: string;
  messageVersion?: string;
  sessionAttributes?: Record<string, string>;
  promptSessionAttributes?: Record<string, string>;
  requestBody?: {
    content?: {
      "application/json"?: {
        properties?: Record<string, unknown>;
      };
    };
  };
}

export interface BedrockActionResponse {
  messageVersion: string;
  response: {
    actionGroup: string;
    apiPath: string;
    httpMethod: string;
    httpStatusCode: number;
    responseBody: {
      "application/json": {
        body: string;
      };
    };
  };
  sessionAttributes?: Record<string, string>;
  promptSessionAttributes?: Record<string, string>;
}

export const getJsonProps = <T extends object = Record<string, unknown>>(
  event: BedrockActionEvent,
): T => (event.requestBody?.content?.["application/json"]?.properties ?? {}) as T;

export const createJsonResponse = (
  event: BedrockActionEvent,
  body: unknown,
  httpStatusCode = 200,
): BedrockActionResponse => ({
  messageVersion: event.messageVersion ?? "1.0",
  response: {
    actionGroup: event.actionGroup,
    apiPath: event.apiPath,
    httpMethod: event.httpMethod,
    httpStatusCode,
    responseBody: {
      "application/json": {
        body: JSON.stringify(body),
      },
    },
  },
  sessionAttributes: event.sessionAttributes,
  promptSessionAttributes: event.promptSessionAttributes,
});
