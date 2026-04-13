import type { Metadata } from "next";
import { Roboto } from "next/font/google";
import { ThemeProvider } from "@aws-bedrock-multiagents/ui";
import Box from "@mui/material/Box";
import Sidebar from "@/components/Sidebar";
import "./globals.css";

const roboto = Roboto({
  weight: ["300", "400", "500", "700"],
  subsets: ["latin"],
  display: "swap",
  variable: "--font-roboto",
});

export const metadata: Metadata = {
  title: "AWS Bedrock Multi-Agent Dashboard",
  description: "Monitor and interact with AWS Bedrock multi-agent system",
};

const SIDEBAR_WIDTH = 240;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={roboto.variable}>
        <ThemeProvider>
          <Box sx={{ display: "flex", minHeight: "100vh" }}>
            <Sidebar width={SIDEBAR_WIDTH} />
            <Box
              component="main"
              sx={{
                flexGrow: 1,
                ml: `${SIDEBAR_WIDTH}px`,
                p: 3,
                bgcolor: "background.default",
                minHeight: "100vh",
              }}
            >
              {children}
            </Box>
          </Box>
        </ThemeProvider>
      </body>
    </html>
  );
}
