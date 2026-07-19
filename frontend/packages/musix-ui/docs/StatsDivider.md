---
category: Layout
---

# StatsDivider

Подписанный шов между секциями: светящаяся точка + подпись + градиентные линии
в обе стороны. Цвет задаётся `hue` (oklch-тон: 275 — вкус, 165 — острова, 75 — сборка).
Классы `.rec-div*` из styles.css.

```tsx
import { StatsDivider } from 'musix-ui';

<StatsDivider label="ПРОСЛУШИВАНИЯ" hue={145} />
<StatsDivider label="КОЛЛЕКЦИЯ" hue={275} />
```
