---
category: Media
---

# BrandMark

Логотип MusiX: тёмная скруглённая плашка с пятью EQ-столбиками фирменных цветов
(фиолетовые → розовый → золотой). Плашка всегда тёмная; `isDark` только подстраивает
тень под фон.

```tsx
import { BrandMark } from 'musix-ui';

<div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
  <BrandMark size={34} isDark />
  <span style={{ fontWeight: 700, fontSize: 18 }}>MusiX</span>
</div>
```
