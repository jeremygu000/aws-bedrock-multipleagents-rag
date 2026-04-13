"use client";

import { useState, useRef, useEffect } from "react";
import Box from "@mui/material/Box";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";
import CircularProgress from "@mui/material/CircularProgress";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import IconButton from "@mui/material/IconButton";
import { sendMessage } from "../actions";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

const SESSION_ID = `session-${Date.now()}`;

export default function ChatContainer() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = async () => {
    const trimmed = inputValue.trim();
    if (!trimmed || isLoading) {
      return;
    }

    const userMessage: ChatMessage = {
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await sendMessage(trimmed, SESSION_ID);
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch {
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, something went wrong. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <AppBar position="static" elevation={1}>
        <Toolbar>
          <IconButton
            component="a"
            href="/"
            color="inherit"
            edge="start"
            sx={{ mr: 1 }}
            aria-label="Back to Dashboard"
          >
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Chat with Agent
          </Typography>
        </Toolbar>
      </AppBar>

      <Paper
        elevation={0}
        sx={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          bgcolor: "grey.50",
          borderRadius: 0,
        }}
      >
        {messages.length === 0 && !isLoading && (
          <Box
            sx={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              p: 4,
            }}
          >
            <Typography variant="body1" color="text.secondary">
              Start a conversation with the Bedrock Agent
            </Typography>
          </Box>
        )}

        {messages.length > 0 && (
          <Box sx={{ pt: 2, pb: 1 }}>
            {messages.map((msg, index) => (
              <ChatMessage key={index} message={msg} />
            ))}
          </Box>
        )}

        {isLoading && (
          <Box sx={{ display: "flex", justifyContent: "flex-start", px: 2, pb: 1 }}>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                bgcolor: "grey.100",
                borderRadius: "18px 18px 18px 4px",
                px: 2,
                py: 1.25,
              }}
            >
              <CircularProgress size={16} thickness={5} />
              <Typography variant="body2" color="text.secondary">
                Thinking…
              </Typography>
            </Box>
          </Box>
        )}

        <div ref={bottomRef} />
      </Paper>

      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSubmit={handleSubmit}
        isLoading={isLoading}
      />
    </Box>
  );
}
