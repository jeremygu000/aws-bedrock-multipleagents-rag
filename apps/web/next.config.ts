import type { NextConfig } from "next";

/* eslint-disable n/no-process-env -- Next.js config requires process.env for zone URLs */
const chatUrl = process.env.CHAT_URL ?? "http://localhost:3001";
const uploadUrl = process.env.UPLOAD_URL ?? "http://localhost:3002";
/* eslint-enable n/no-process-env */

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/chat",
        destination: `${chatUrl}/chat`,
      },
      {
        source: "/chat/:path*",
        destination: `${chatUrl}/chat/:path*`,
      },
      {
        source: "/upload",
        destination: `${uploadUrl}/upload`,
      },
      {
        source: "/upload/:path*",
        destination: `${uploadUrl}/upload/:path*`,
      },
    ];
  },
};

export default nextConfig;
