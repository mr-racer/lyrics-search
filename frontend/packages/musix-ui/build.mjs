// Library build for musix-ui: one ESM entry, react externalized.
// esbuild resolves from frontend/node_modules (vite's dependency).
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

await build({
  entryPoints: [join(here, 'src/index.jsx')],
  bundle: true,
  format: 'esm',
  outfile: join(here, 'dist/index.es.js'),
  external: ['react', 'react-dom', 'react/jsx-runtime'],
  jsx: 'automatic',
  loader: { '.jsx': 'jsx' },
  logLevel: 'info',
});
