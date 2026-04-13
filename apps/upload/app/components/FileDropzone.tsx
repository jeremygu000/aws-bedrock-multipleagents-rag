"use client";

import { useRef, useState, type DragEvent, type ChangeEvent } from "react";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

interface FileDropzoneProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

const ACCEPTED_EXTENSIONS = ".pdf,.txt,.md,.docx,.csv";

export default function FileDropzone({ onFilesSelected, disabled = false }: FileDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    if (!disabled) {
      setIsDragOver(true);
    }
  }

  function handleDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(false);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) {
      return;
    }
    const droppedFiles = Array.from(e.dataTransfer.files);
    if (droppedFiles.length > 0) {
      onFilesSelected(droppedFiles);
    }
  }

  function handleClick() {
    if (!disabled) {
      inputRef.current?.click();
    }
  }

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(e.target.files ?? []);
    if (selectedFiles.length > 0) {
      onFilesSelected(selectedFiles);
    }
    e.target.value = "";
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPTED_EXTENSIONS}
        onChange={handleInputChange}
        style={{ display: "none" }}
        aria-label="File input"
      />
      <Paper
        variant="outlined"
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        sx={{
          p: 6,
          textAlign: "center",
          cursor: disabled ? "not-allowed" : "pointer",
          border: "2px dashed",
          borderColor: isDragOver ? "primary.main" : "divider",
          bgcolor: isDragOver ? "action.hover" : "background.paper",
          transition: "border-color 0.2s ease, background-color 0.2s ease",
          opacity: disabled ? 0.6 : 1,
          "&:hover": disabled
            ? {}
            : {
                borderColor: "primary.light",
                bgcolor: "action.hover",
              },
        }}
      >
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1.5 }}>
          <CloudUploadIcon
            sx={{
              fontSize: 56,
              color: isDragOver ? "primary.main" : "action.active",
              transition: "color 0.2s ease",
            }}
          />
          <Typography variant="h6" component="p" color={isDragOver ? "primary" : "text.primary"}>
            {isDragOver ? "Release to upload" : "Drag & drop files here or click to browse"}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Supported: PDF, TXT, MD, DOCX, CSV
          </Typography>
        </Box>
      </Paper>
    </>
  );
}
