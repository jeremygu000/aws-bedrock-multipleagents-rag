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
        properties?: BedrockProperty[] | Record<string, unknown>;
      };
    };
  };
}

interface BedrockProperty {
  name: string;
  type: string;
  value: string;
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

const normalizeProperties = (
  raw: BedrockProperty[] | Record<string, unknown> | undefined,
): Record<string, unknown> => {
  if (!raw) {
    return {};
  }
  if (!Array.isArray(raw)) {
    return raw;
  }
  const out: Record<string, unknown> = {};
  for (const item of raw) {
    if (item.name) {
      let value: unknown = item.value;
      const t = (item.type ?? "string").toLowerCase();
      if (t === "integer" || t === "number") {
        const n = Number(value);
        if (!Number.isNaN(n)) {
          value = n;
        }
      } else if (t === "boolean") {
        value = String(value).toLowerCase() === "true";
      }
      out[item.name] = value;
    }
  }
  return out;
};

export const getJsonProps = <T extends object = Record<string, unknown>>(
  event: BedrockActionEvent,
): T => normalizeProperties(event.requestBody?.content?.["application/json"]?.properties) as T;

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
