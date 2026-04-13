import type { ChatMessage as ChatMessageType } from "./ChatContainer";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";

interface ChatMessageProps {
  message: ChatMessageType;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  const formattedTime = message.timestamp.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        mb: 1.5,
        px: 1,
      }}
    >
      <Box sx={{ maxWidth: "72%", minWidth: 80 }}>
        <Paper
          elevation={1}
          sx={{
            px: 2,
            py: 1.25,
            borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
            bgcolor: isUser ? "primary.main" : "grey.100",
            color: isUser ? "primary.contrastText" : "text.primary",
          }}
        >
          <Typography variant="body1" sx={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {message.content}
          </Typography>
        </Paper>
        <Typography
          variant="caption"
          sx={{
            display: "block",
            mt: 0.25,
            color: "text.disabled",
            textAlign: isUser ? "right" : "left",
            px: 0.5,
          }}
        >
          {formattedTime}
        </Typography>
      </Box>
    </Box>
  );
}
