import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Tier 0: Vite build around the existing single-file frontend.
// - root is this directory (frontend/), entry is ./index.html → ./src/main.jsx
// - build output goes to ./dist, which FastAPI serves as the SPA.
// - dev server proxies /api → uvicorn on :8000. Every backend call (API,
//   /covers, audio streams) goes through the /api/v1 prefix, so one rule covers
//   everything. Run `uvicorn` and `npm run dev` side by side for HMR.
export default defineConfig({
  plugins: [
    react(),
    // PWA (см. docs/superpowers/specs/2026-07-10-pwa-packaging-design.md).
    // Ключевой инвариант: /api/** service worker'ом НЕ перехватывается и не
    // кэшируется — аудио-стрим (Range, ?st= токены) идёт мимо SW.
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['apple-touch-icon.png'],
      manifest: {
        name: 'MusiX',
        short_name: 'MusiX',
        description: 'Локальная музыка: плеер, семантический поиск, ИИ-подборки',
        lang: 'ru',
        start_url: '/',
        scope: '/',
        display: 'standalone',
        background_color: '#0d0d10',
        theme_color: '#0d0d10',
        icons: [
          { src: '/pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: '/pwa-512x512.png', sizes: '512x512', type: 'image/png' },
          { src: '/pwa-512x512-maskable.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,png,svg,ico}'],
        navigateFallback: 'index.html',
        navigateFallbackDenylist: [/^\/api\//, /^\/docs/, /^\/covers\//],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/fonts\.(googleapis|gstatic)\.com\/.*/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts',
              expiration: { maxEntries: 30, maxAgeSeconds: 60 * 60 * 24 * 365 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
    }),
  ],
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
