---
category: Media
---

# MosaicCover

Обложка плейлиста: мозаика из первых 1–4 обложек треков (2×2; при трёх — первая
на всю ширину). Пустой плейлист — градиент с нотой ♫. `size` число (px) или
CSS-строка ('100%' — квадратность держит `aspect-ratio`).

```tsx
import { MosaicCover } from 'musix-ui';

<MosaicCover trackIds={[1,2,3,4]} coverPaths={[p1,p2,p3,p4]} size={160} />
<MosaicCover trackIds={[]} coverPaths={[]} size={160} />   // пустой плейлист
```
