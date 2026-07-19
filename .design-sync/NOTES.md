# design-sync NOTES — musix-ui

## Как собирать
- Библиотека: `node frontend/packages/musix-ui/build.mjs` (esbuild из frontend/node_modules; react external).
- Конвертер/драйвер из корня репо:
  `node .ds-sync/resync.mjs --config .design-sync/config.json --node-modules ./frontend/node_modules --entry ./frontend/packages/musix-ui/dist/index.es.js --out ./ds-bundle --no-render-check [--remote .design-sync/.cache/remote-sync.json]`
- **`--no-render-check` обязателен на этой машине**: playwright/chromium не установлены — владелец отказался от установки (2026-07-19) и верифицирует карточки сам через `.review.html` (`node .ds-sync/storybook/http-serve.mjs ./ds-bundle`). Capture/грейды не запускались ни разу — carry-forward-грейдов нет; при появлении playwright прогнать полный validate+capture.

## Особенности пакета
- Код — чистый JSX; **`frontend/packages/musix-ui/index.d.ts` ведётся вручную** и является источником пропсов для карточек. Меняешь пропсы компонента — обнови index.d.ts (иначе контракт для дизайн-агента разъедется).
- `styles.css` живёт В ПАКЕТЕ (перенесён из frontend/src/index.css); приложение импортирует его через однострочный shim `frontend/src/index.css`. cssEntry ограничен корнем пакета — наружу указывать нельзя.
- Шрифты — remote Google Fonts `@import` первой строкой styles.css → validate печатает `[FONT_REMOTE]` (норма).
- `guidelinesGlob: []` — иначе дефолтный glob утаскивает docs/*.md (пер-компонентные доки) в guidelines/ дублями.
- Группировка карточек — frontmatter `category:` в `frontend/packages/musix-ui/docs/<Name>.md` (Controls/Media/Layout/Data).

## Known render warns
- `[TOKENS_MISSING] 12 CSS custom properties` (--lyric-*, --ob-glass-*, --ob-blob*, --ob-card-*) — задаются на рантайме инлайн-стилями экранов приложения; в превью GlassCard/FeatureCard/ModeCard ставятся на обёртке. Не чинить.
- `[FONT_REMOTE]` Geist/Playfair/JetBrains Mono/Noto — remote-шрифты, норма.
- `[RENDER_SKIPPED]` — см. выше, осознанный выбор владельца.

## Re-sync risks
- Ручной index.d.ts может отстать от src/index.jsx (нет typecheck-связки).
- Превью HintBadge/InfoTip показывают закрытое состояние (тултипы — hover-only, статически не рендерятся).
- CrossfadeText-превью живое (setInterval) — скриншот в любой момент валиден, но кадры могут отличаться.
- Рендеры НИКОГДА не проверялись машиной — только глазами владельца (2026-07-19, все 26 одобрены).
- До первого синка в рабочем дереве уже были незакоммиченные правки (.gitignore, package.json, main.jsx…) — возможно, след прежней прерванной попытки или WIP владельца; при коммите разделять аккуратно.
