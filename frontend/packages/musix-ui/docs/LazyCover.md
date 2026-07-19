---
category: Media
---

# LazyCover

Лениво загружаемая обложка по прямому URL (`<img loading="lazy">` — CSS-фоны грузятся
жадно даже в скрытых секциях, поэтому списочный арт идёт через этот компонент).
Без `url` — фирменный градиент-заглушка; ошибка загрузки тоже откатывается на градиент.

```tsx
import { LazyCover } from 'musix-ui';

<LazyCover url={coverUrl} style={{ width: 46, height: 46, borderRadius: 10 }} />
<LazyCover style={{ width: 46, height: 46, borderRadius: 10 }} />  // заглушка
```
