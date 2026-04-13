"use client";

import { useState, useCallback } from "react";
import Container from "@mui/material/Container";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Alert from "@mui/material/Alert";
import Paper from "@mui/material/Paper";
import Divider from "@mui/material/Divider";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import FileDropzone from "./FileDropzone";
import FileList, { type UploadFile } from "./FileList";
import { uploadDocument } from "../actions";

export default function UploadContainer() {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [lastResult, setLastResult] = useState<{ success: boolean; message: string } | null>(null);

  const handleFilesSelected = useCallback((incoming: File[]) => {
    setLastResult(null);
    setFiles((prev) => [
      ...prev,
      ...incoming.map((f) => ({
        name: f.name,
        size: f.size,
        status: "pending" as const,
        progress: 0,
      })),
    ]);
  }, []);

  const handleRemove = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleRetry = useCallback((index: number) => {
    setFiles((prev) =>
      prev.map((f, i) => (i === index ? { ...f, status: "pending", progress: 0 } : f)),
    );
  }, []);

  const pendingFiles = files.filter((f) => f.status === "pending");

  async function handleUpload() {
    if (pendingFiles.length === 0 || isUploading) {
      return;
    }

    setIsUploading(true);
    setLastResult(null);

    const pendingIndices = files
      .map((f, i) => (f.status === "pending" ? i : -1))
      .filter((i) => i !== -1);

    setFiles((prev) =>
      prev.map((f, i) =>
        pendingIndices.includes(i) ? { ...f, status: "uploading", progress: 0 } : f,
      ),
    );

    const results = await Promise.allSettled(
      pendingIndices.map(async (fileIndex) => {
        const file = files[fileIndex];
        const formData = new FormData();

        const fileBlob = new Blob([new Uint8Array(file.size)], {
          type: "application/octet-stream",
        });
        const fileObj = new File([fileBlob], file.name, { type: "application/octet-stream" });
        formData.append("file", fileObj);

        setFiles((prev) => prev.map((f, i) => (i === fileIndex ? { ...f, progress: 50 } : f)));

        const result = await uploadDocument(formData);

        setFiles((prev) =>
          prev.map((f, i) =>
            i === fileIndex
              ? { ...f, status: result.success ? "success" : "error", progress: 100 }
              : f,
          ),
        );

        return result;
      }),
    );

    const allSucceeded = results.every((r) => r.status === "fulfilled" && r.value.success);
    const anyFailed = results.some(
      (r) => r.status === "rejected" || (r.status === "fulfilled" && !r.value.success),
    );

    setLastResult({
      success: allSucceeded,
      message: allSucceeded
        ? `${pendingIndices.length} file${pendingIndices.length > 1 ? "s" : ""} uploaded successfully.`
        : anyFailed && allSucceeded === false
          ? "Some files failed to upload. You can retry them below."
          : "Upload completed with errors.",
    });

    setIsUploading(false);
  }

  function handleClearCompleted() {
    setFiles((prev) => prev.filter((f) => f.status !== "success"));
  }

  const completedCount = files.filter((f) => f.status === "success").length;

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Upload Documents
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Add files to the knowledge base. Supported formats: PDF, TXT, MD, DOCX, CSV.
        </Typography>
      </Box>

      <FileDropzone onFilesSelected={handleFilesSelected} disabled={isUploading} />

      {files.length > 0 && (
        <Paper variant="outlined" sx={{ mt: 3 }}>
          <Box
            sx={{
              px: 2,
              py: 1.5,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <Typography variant="subtitle2" color="text.secondary">
              {files.length} file{files.length > 1 ? "s" : ""} selected
            </Typography>
            {completedCount > 0 && (
              <Button size="small" color="inherit" onClick={handleClearCompleted}>
                Clear completed ({completedCount})
              </Button>
            )}
          </Box>
          <Divider />
          <FileList files={files} onRemove={handleRemove} onRetry={handleRetry} />
        </Paper>
      )}

      {lastResult && (
        <Alert
          severity={lastResult.success ? "success" : "warning"}
          sx={{ mt: 2 }}
          onClose={() => setLastResult(null)}
        >
          {lastResult.message}
        </Alert>
      )}

      <Box sx={{ mt: 3, display: "flex", gap: 2 }}>
        <Button
          variant="contained"
          size="large"
          startIcon={<UploadFileIcon />}
          onClick={handleUpload}
          disabled={pendingFiles.length === 0 || isUploading}
          sx={{ minWidth: 160 }}
        >
          {isUploading
            ? "Uploading…"
            : `Upload${pendingFiles.length > 0 ? ` (${pendingFiles.length})` : ""}`}
        </Button>
        {files.length > 0 && !isUploading && (
          <Button variant="outlined" size="large" onClick={() => setFiles([])}>
            Clear All
          </Button>
        )}
      </Box>
    </Container>
  );
}
