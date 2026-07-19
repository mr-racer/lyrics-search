---
category: Media
---

# ResultCard

Строка результата поиска: обложка (AlbumCover), название/артист/год, справа бейдж
«NN% · режим» (lyrics — фиолетовый, audio — розово-фиолетовый, hybrid — янтарный)
и жанр. `onPlay` добавляет круглую кнопку ▶. Требует объект палитры `c` из `useColors(isDark)`.

```tsx
import { ResultCard, useColors } from 'musix-ui';

const c = useColors(isDark);
<ResultCard
  isDark={isDark} c={c}
  hit={{ score: 0.87, matched_on: 'lyrics',
         track: { title: 'Группа крови', artist: 'Кино', year: 1988, genre: 'post-punk' } }}
  onClick={openTrack} onPlay={playTrack}
/>
```
