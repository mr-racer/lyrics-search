---
category: Data
---

# Sparkline

Мини-график активности за последние 14 дней: линия + градиентная заливка.
`days` — `[{date:'YYYY-MM-DD', count}]`, пропущенные дни считаются нулями;
`hue` — oklch-тон (145 — зелёный «активность»).

```tsx
import { Sparkline } from 'musix-ui';

<Sparkline hue={145} days={[
  { date: '2026-07-18', count: 24 },
  { date: '2026-07-17', count: 11 },
  { date: '2026-07-15', count: 32 },
]} />
```
