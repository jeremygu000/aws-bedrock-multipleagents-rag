import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/handler.ts"],
  format: ["cjs"],
  target: "node20",
  platform: "node",
  sourcemap: true,
  clean: true,
  outDir: "dist",
  outExtension: () => ({
    js: ".cjs",
  }),
});
