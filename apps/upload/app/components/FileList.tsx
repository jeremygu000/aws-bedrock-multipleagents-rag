"use client";

import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import ListItemSecondaryAction from "@mui/material/ListItemSecondaryAction";
import IconButton from "@mui/material/IconButton";
import Chip from "@mui/material/Chip";
import LinearProgress from "@mui/material/LinearProgress";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import DeleteIcon from "@mui/icons-material/Delete";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import ReplayIcon from "@mui/icons-material/Replay";

export interface UploadFile {
  name: string;
  size: number;
  status: "pending" | "uploading" | "success" | "error";
  progress: number;
}

interface FileListProps {
  files: UploadFile[];
  onRemove: (index: number) => void;
  onRetry: (index: number) => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const STATUS_CHIP_MAP = {
  pending: { label: "Pending", color: "default" as const },
  uploading: { label: "Uploading…", color: "primary" as const },
  success: { label: "Uploaded", color: "success" as const },
  error: { label: "Failed", color: "error" as const },
};

export default function FileList({ files, onRemove, onRetry }: FileListProps) {
  if (files.length === 0) {
    return null;
  }

  return (
    <List disablePadding>
      {files.map((file, index) => {
        const chip = STATUS_CHIP_MAP[file.status];
        return (
          <ListItem
            key={`${file.name}-${index}`}
            divider={index < files.length - 1}
            sx={{ flexDirection: "column", alignItems: "stretch", pr: 10 }}
          >
            <Box sx={{ display: "flex", alignItems: "center", width: "100%", gap: 1 }}>
              {file.status === "success" && <CheckCircleIcon fontSize="small" color="success" />}
              {file.status === "error" && <ErrorIcon fontSize="small" color="error" />}
              <ListItemText
                primary={
                  <Typography variant="body2" noWrap sx={{ maxWidth: "55ch" }}>
                    {file.name}
                  </Typography>
                }
                secondary={formatFileSize(file.size)}
                sx={{ m: 0 }}
              />
              <Chip label={chip.label} color={chip.color} size="small" sx={{ ml: "auto" }} />
            </Box>

            {file.status === "uploading" && (
              <LinearProgress
                variant={file.progress > 0 ? "determinate" : "indeterminate"}
                value={file.progress}
                sx={{ mt: 1, borderRadius: 1 }}
              />
            )}

            <ListItemSecondaryAction>
              {file.status === "error" && (
                <IconButton
                  edge="end"
                  size="small"
                  aria-label="retry upload"
                  onClick={() => onRetry(index)}
                  sx={{ mr: 0.5 }}
                >
                  <ReplayIcon fontSize="small" />
                </IconButton>
              )}
              <IconButton
                edge="end"
                size="small"
                aria-label="remove file"
                onClick={() => onRemove(index)}
                disabled={file.status === "uploading"}
              >
                <DeleteIcon fontSize="small" />
              </IconButton>
            </ListItemSecondaryAction>
          </ListItem>
        );
      })}
    </List>
  );
}
