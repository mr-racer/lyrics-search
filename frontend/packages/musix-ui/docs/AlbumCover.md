---
category: Media
---

# AlbumCover

Обложка альбома. С `coverPath` — `<img>` с ленивой загрузкой и серверной миниатюрой
(`?w=320`); без него (или при ошибке) — детерминированный градиент от `title`/`artist`
с двумя инициалами. Радиус подбирается от размера, `radius` переопределяет.
`fluid` растягивает на 100% контейнера, `eager` — полноразмерный арт с синхронной
отрисовкой (hero-обложка плеера).

```tsx
import { AlbumCover } from 'musix-ui';

<AlbumCover title="Группа крови" artist="Кино" size={50} isDark />
<AlbumCover title="Мракобесие и джаз" artist="Агата Кристи" size={160} isDark />
<AlbumCover fluid title="…" artist="…" isDark />   // в сетке с aspect-ratio: 1/1
```
