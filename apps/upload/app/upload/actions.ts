"use server";

export async function uploadDocument(
  formData: FormData,
): Promise<{ success: boolean; message: string; documentId?: string }> {
  // TODO: Replace with actual S3 upload + ingestion pipeline trigger
  const file = formData.get("file") as File;
  await new Promise((resolve) => setTimeout(resolve, 2000));
  return {
    success: true,
    message: `File "${file.name}" uploaded successfully. Ingestion pipeline triggered.`,
    documentId: `doc-${Date.now()}`,
  };
}
