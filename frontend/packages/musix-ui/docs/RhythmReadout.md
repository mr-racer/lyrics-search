---
category: Data
---

# RhythmReadout

Стеклянный чип-показание для шапок секций: эмодзи-иконка, крупное значение,
tracked-caps подпись цвета `hue`, опциональная третья строка `foot`.
`grow` — flex-растёт в ряду (1 1 280px).

```tsx
import { RhythmReadout } from 'musix-ui';

<div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
  <RhythmReadout icon="🔥" value="12 дней" label="Серия" foot="лучшая — 21 день" hue={30} isDark />
  <RhythmReadout icon="🎧" value="Пятница" label="Самый активный день" hue={275} isDark grow />
</div>
```
