import type { OpenNextConfig } from "@opennextjs/aws/types/open-next";

const config = {
  default: {
    install: {
      packages: ["@swc/helpers@0.5.17", "styled-jsx@5.1.6", "@next/env"],
      arch: "arm64",
    },
  },
  // Next.js build is done externally via `next build`
  buildCommand: "exit 0",
  // Root directory of the pnpm monorepo (where root package.json lives)
  packageJsonPath: "../../",
} satisfies OpenNextConfig;

export default config;
