type LogContext = Record<string, unknown>;

const formatLog = (level: "INFO" | "ERROR", message: string, context: LogContext) =>
  JSON.stringify({
    level,
    message,
    ...context,
  });

export const logInfo = (message: string, context: LogContext = {}): void => {
  console.log(formatLog("INFO", message, context));
};

export const logError = (message: string, context: LogContext = {}): void => {
  console.error(formatLog("ERROR", message, context));
};
