import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Grid from "@mui/material/Grid2";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardActions from "@mui/material/CardActions";
import Button from "@mui/material/Button";
import ChatIcon from "@mui/icons-material/Chat";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import BarChartIcon from "@mui/icons-material/BarChart";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import GroupIcon from "@mui/icons-material/Group";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";

const QUICK_ACTIONS = [
  {
    title: "Chat with Agent",
    description:
      "Start a conversation with the AWS Bedrock supervisor agent for APRA AMCOS queries and work search.",
    icon: <ChatIcon fontSize="large" color="primary" />,
    href: "/chat",
    buttonLabel: "Open Chat",
    crossZone: true,
  },
  {
    title: "Upload Documents",
    description: "Upload documents to the RAG knowledge base for indexing and hybrid retrieval.",
    icon: <UploadFileIcon fontSize="large" color="secondary" />,
    href: "/upload",
    buttonLabel: "Upload Files",
    crossZone: true,
  },
  {
    title: "View Monitoring",
    description:
      "Inspect agent call volume, response times, RAG pipeline metrics, and success rates.",
    icon: <BarChartIcon fontSize="large" sx={{ color: "success.main" }} />,
    href: "/monitoring",
    buttonLabel: "View Metrics",
    crossZone: false,
  },
];

const STATS = [
  {
    label: "Total Queries",
    value: "—",
    icon: <TrendingUpIcon color="primary" />,
  },
  {
    label: "Active Sessions",
    value: "—",
    icon: <GroupIcon color="secondary" />,
  },
  {
    label: "Success Rate",
    value: "—",
    icon: <CheckCircleIcon sx={{ color: "success.main" }} />,
  },
];

export default function DashboardPage() {
  return (
    <Box sx={{ maxWidth: 1100, mx: "auto" }}>
      <Typography variant="h4" fontWeight={700} gutterBottom>
        AWS Bedrock Multi-Agent Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Manage your multi-agent system, monitor performance, and interact with the Bedrock agents.
      </Typography>

      <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
        Quick Actions
      </Typography>
      <Grid container spacing={3} sx={{ mb: 5 }}>
        {QUICK_ACTIONS.map((action) => (
          <Grid key={action.title} size={{ xs: 12, sm: 6, md: 4 }}>
            <Card
              variant="outlined"
              sx={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                transition: "box-shadow 0.2s",
                "&:hover": { boxShadow: 3 },
              }}
            >
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ mb: 1.5 }}>{action.icon}</Box>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  {action.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {action.description}
                </Typography>
              </CardContent>
              <CardActions sx={{ px: 2, pb: 2 }}>
                {action.crossZone ? (
                  <Button
                    component="a"
                    href={action.href}
                    variant="contained"
                    size="small"
                    disableElevation
                  >
                    {action.buttonLabel}
                  </Button>
                ) : (
                  <Button
                    component="a"
                    href={action.href}
                    variant="contained"
                    size="small"
                    disableElevation
                  >
                    {action.buttonLabel}
                  </Button>
                )}
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
        Overview
      </Typography>
      <Grid container spacing={3}>
        {STATS.map((stat) => (
          <Grid key={stat.label} size={{ xs: 12, sm: 4 }}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                  {stat.icon}
                  <Box>
                    <Typography variant="h5" fontWeight={700}>
                      {stat.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {stat.label}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
