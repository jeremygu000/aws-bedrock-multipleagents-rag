import type { Metadata } from "next";
import { Roboto } from "next/font/google";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { ThemeProvider } from "@aws-bedrock-multiagents/ui";
import "./globals.css";

const roboto = Roboto({
  weight: ["300", "400", "500", "700"],
  subsets: ["latin"],
  display: "swap",
  variable: "--font-roboto",
});

export const metadata: Metadata = {
  title: "Document Upload",
  description: "Upload and ingest documents into the knowledge base",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={roboto.variable}>
        <ThemeProvider>
          <AppBar position="static" elevation={1}>
            <Toolbar>
              <IconButton
                component="a"
                href="/"
                edge="start"
                color="inherit"
                aria-label="back to dashboard"
                sx={{ mr: 1 }}
              >
                <ArrowBackIcon />
              </IconButton>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Document Upload
              </Typography>
            </Toolbar>
          </AppBar>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
