import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Tier 0: Vite build around the existing single-file frontend.
// - root is this directory (frontend/), entry is ./index.html → ./src/main.jsx
// - build output goes to ./dist, which FastAPI serves as the SPA.
// - dev server proxies /api → uvicorn on :8000. Every backend call (API,
//   /covers, audio streams) goes through the /api/v1 prefix, so one rule covers
//   everything. Run `uvicorn` and `npm run dev` side by side for HMR.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
});
