import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  basePath: "/chat",
  assetPrefix: "/chat",
};

export default nextConfig;
