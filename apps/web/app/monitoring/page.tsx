import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Grid from "@mui/material/Grid2";
import Paper from "@mui/material/Paper";
import BarChartIcon from "@mui/icons-material/BarChart";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import AccountTreeIcon from "@mui/icons-material/AccountTree";

const METRIC_CARDS = [
  {
    title: "Agent Call Volume",
    description:
      "Total number of agent invocations over time. Connect to CloudWatch to populate live data.",
    icon: <BarChartIcon fontSize="large" color="primary" />,
  },
  {
    title: "Response Time",
    description: "P50 / P95 / P99 latency for supervisor and specialist agent round-trips.",
    icon: <AccessTimeIcon fontSize="large" color="secondary" />,
  },
  {
    title: "Success Rate",
    description:
      "Percentage of agent calls that returned a valid response without guardrail blocks.",
    icon: <CheckCircleIcon fontSize="large" sx={{ color: "success.main" }} />,
  },
  {
    title: "RAG Pipeline Metrics",
    description:
      "Hybrid retrieval recall, reranking scores, and HyDE / CRAG / self-reflection usage rates.",
    icon: <AccountTreeIcon fontSize="large" sx={{ color: "warning.main" }} />,
  },
];

export default function MonitoringPage() {
  return (
    <Box sx={{ maxWidth: 1100, mx: "auto" }}>
      <Typography variant="h4" fontWeight={700} gutterBottom>
        Agent Monitoring
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Real-time metrics for your AWS Bedrock multi-agent system. Connect CloudWatch or Grafana to
        populate live data.
      </Typography>

      <Grid container spacing={3}>
        {METRIC_CARDS.map((card) => (
          <Grid key={card.title} size={{ xs: 12, sm: 6 }}>
            <Paper
              variant="outlined"
              sx={{
                p: 3,
                height: "100%",
                display: "flex",
                flexDirection: "column",
                gap: 2,
                transition: "box-shadow 0.2s",
                "&:hover": { boxShadow: 2 },
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                {card.icon}
                <Typography variant="h6" fontWeight={600}>
                  {card.title}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {card.description}
              </Typography>
              <Box
                sx={{
                  flexGrow: 1,
                  minHeight: 120,
                  borderRadius: 1,
                  bgcolor: "action.hover",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Typography variant="caption" color="text.disabled">
                  Chart placeholder — wire up CloudWatch metrics
                </Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
