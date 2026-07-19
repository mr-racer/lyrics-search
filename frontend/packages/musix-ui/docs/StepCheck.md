---
category: Controls
---

# StepCheck

Маленькая SVG-галочка (`stroke: currentColor`) для завершённых шагов мастеров
и списков «сделано». Цвет наследуется от родителя.

```tsx
import { StepCheck } from 'musix-ui';

<span style={{ color: 'oklch(63% 0.17 142)', display: 'inline-flex', gap: 6, alignItems: 'center' }}>
  <StepCheck size={12} /> Библиотека проиндексирована
</span>
```
