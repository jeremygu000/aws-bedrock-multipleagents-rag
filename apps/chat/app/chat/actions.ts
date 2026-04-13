"use server";

export async function sendMessage(prompt: string, _sessionId: string): Promise<string> {
  // TODO: Replace with Bedrock Agent Runtime SDK call
  // Simulate delay for now
  await new Promise((resolve) => setTimeout(resolve, 1000));
  return `This is a placeholder response to: "${prompt}". Connect Bedrock Agent Runtime SDK here.`;
}
