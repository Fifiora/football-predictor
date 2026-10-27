import { defineConfig } from "astro/config";

// Project pages on GitHub Pages live under /<repo>/; set SITE_BASE in CI.
export default defineConfig({
  base: process.env.SITE_BASE ?? "/",
  output: "static",
});
