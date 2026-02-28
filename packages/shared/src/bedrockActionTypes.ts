export interface BedrockActionParameter {
  name: string;
  type?: string;
  value: string;
}

export interface BedrockActionEvent {
  messageVersion: string;
  inputText?: string;
  apiPath?: string;
  actionGroup?: string;
  httpMethod?: string;
  parameters?: BedrockActionParameter[];
  sessionAttributes?: Record<string, string>;
  promptSessionAttributes?: Record<string, string>;
  requestBody?: unknown;
}

export interface BedrockActionResponse {
  messageVersion: string;
  response: {
    actionGroup?: string;
    apiPath?: string;
    httpMethod?: string;
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
