"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Switch from "@mui/material/Switch";
import Divider from "@mui/material/Divider";

interface FeatureFlag {
  key: string;
  label: string;
  description: string;
  defaultEnabled: boolean;
}

const FEATURE_FLAGS: FeatureFlag[] = [
  {
    key: "hyde",
    label: "HyDE Retrieval",
    description:
      "Hypothetical Document Embeddings — generates a hypothetical answer to improve dense retrieval recall.",
    defaultEnabled: true,
  },
  {
    key: "crag",
    label: "CRAG Fallback",
    description:
      "Corrective RAG — grades retrieved documents and falls back to web search when relevance is low.",
    defaultEnabled: false,
  },
  {
    key: "reflection",
    label: "Self-Reflection",
    description:
      "LLM-as-judge faithfulness and relevance grading with adaptive retry on low-confidence answers.",
    defaultEnabled: true,
  },
  {
    key: "decomposition",
    label: "Query Decomposition",
    description:
      "Breaks complex multi-hop questions into sub-questions for parallel retrieval and synthesis.",
    defaultEnabled: false,
  },
  {
    key: "community",
    label: "Community Detection",
    description:
      "Leiden algorithm community detection for hierarchical knowledge graph summarisation (GraphRAG).",
    defaultEnabled: false,
  },
];

export default function SettingsPage() {
  const [flags, setFlags] = useState<Record<string, boolean>>(
    Object.fromEntries(FEATURE_FLAGS.map((f) => [f.key, f.defaultEnabled])),
  );

  function handleToggle(key: string) {
    setFlags((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  return (
    <Box sx={{ maxWidth: 700, mx: "auto" }}>
      <Typography variant="h4" fontWeight={700} gutterBottom>
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Toggle RAG pipeline feature flags. Changes here are local only — update your deployment
        environment variables to persist them.
      </Typography>

      <Paper variant="outlined">
        <List disablePadding>
          {FEATURE_FLAGS.map((flag, index) => (
            <Box key={flag.key}>
              <ListItem
                secondaryAction={
                  <Switch
                    edge="end"
                    checked={flags[flag.key]}
                    onChange={() => handleToggle(flag.key)}
                    inputProps={{ "aria-label": flag.label }}
                  />
                }
                sx={{ py: 2, px: 3 }}
              >
                <ListItemText
                  primary={
                    <Typography variant="body1" fontWeight={500}>
                      {flag.label}
                    </Typography>
                  }
                  secondary={flag.description}
                  secondaryTypographyProps={{ variant: "body2", mt: 0.5 }}
                />
              </ListItem>
              {index < FEATURE_FLAGS.length - 1 && <Divider />}
            </Box>
          ))}
        </List>
      </Paper>
    </Box>
  );
}
