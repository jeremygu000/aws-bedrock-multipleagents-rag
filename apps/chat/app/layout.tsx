import type { ReactNode } from "react";
import type { Metadata } from "next";
import { Roboto } from "next/font/google";
import { ThemeProvider } from "@aws-bedrock-multiagents/ui";
import "./globals.css";

const roboto = Roboto({
  weight: ["300", "400", "500", "700"],
  subsets: ["latin"],
  display: "swap",
  variable: "--font-roboto",
});

export const metadata: Metadata = {
  title: "Chat with Agent",
  description: "Chat interface for AWS Bedrock Agent",
};

interface RootLayoutProps {
  children: ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={roboto.variable}>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
