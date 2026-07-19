---
category: Controls
---

# HintBadge

Значок «i» с подсказкой. На десктопе — hover/focus-тултип (`placement` 'up'|'down'),
на мобильном (≤768px) — тап открывает bottom sheet через портал (значок часто живёт
в строках с overflow-clip). `label` принимает JSX. Стили — классы `hint-badge*` из styles.css.

```tsx
import { HintBadge } from 'musix-ui';

<HintBadge size={18} label="Похожесть считается по CLAP-эмбеддингам звука" />
<HintBadge placement="down" label={<>Точность: <b>bm25 + dense</b> (RRF)</>} />
```
