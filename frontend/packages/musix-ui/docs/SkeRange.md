---
category: Controls
---

# SkeRange

Скеуоморфный слайдер: «вырезанный» жёлоб + рельефная ручка. Нативный `<input type=range>`
лежит прозрачно сверху, поэтому drag, клик по треку, клавиатура и a11y работают из коробки;
видимые заливка и ручка зеркалят значение.

- `accent` — цвет заливки и свечения на конкретном инстансе (фиолетовый, золотой…);
- `animated` — ручка и заливка скользят между фиксированными стопами (для дискретных значений);
- `bipolar` — заливка растёт от нуля в центре (диапазоны −N..+N);
- `disabled` — гасит контрол (opacity + pointer-events).

```tsx
import { SkeRange } from 'musix-ui';

const [v, setV] = useState(64);
<SkeRange value={v} onChange={setV} ariaLabel="Громкость" />
<SkeRange value={2} min={-5} max={5} bipolar animated accent="oklch(78% 0.13 75)" onChange={...} />
```

Светлая тема управляется `body[data-theme="light"]` (перекрашивает жёлоб и ручку).
