---
category: Layout
---

# GlassCard

Стеклянная карточка онбординга (`.ob-glass`): блюр, светящаяся кромка, мягкая тень.
Родитель должен задать CSS-переменные `--ob-glass-bg`, `--ob-glass-edge`, `--ob-glass-sheen`
(в приложении их ставят экраны онбординга; см. пример).

```tsx
import { GlassCard } from 'musix-ui';

<div style={{
  '--ob-glass-bg': 'rgba(26,22,44,.55)',
  '--ob-glass-edge': 'rgba(255,255,255,.14)',
  '--ob-glass-sheen': 'rgba(255,255,255,.06)',
} as React.CSSProperties}>
  <GlassCard style={{ padding: 24 }}>
    <h3>Добро пожаловать в MusiX</h3>
    <p>Локальный плеер с семантическим поиском.</p>
  </GlassCard>
</div>
```
