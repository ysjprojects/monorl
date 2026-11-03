import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.MONORL_API ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        // One vendor chunk: splitting recharts from react creates a chunk cycle where the charts
        // chunk evaluates before React is initialised (`forwardRef` of undefined) in production.
        manualChunks: (id: string) => (id.includes("node_modules") ? "vendor" : undefined),
      },
    },
  },
});
