---
category: Data
---

# SkeuoArcGauge

Полукруглый шкальный индикатор (0..1): светящаяся дуга цвета `hue`, точка-каретка
на конце, крупный процент под дугой, tracked-caps подпись `label` и пояснение `sub`.
Анимируется при появлении в вьюпорте (IntersectionObserver), уважает reduced-motion.

```tsx
import { SkeuoArcGauge } from 'musix-ui';

<SkeuoArcGauge value={0.78} hue={145} label="Дослушивание" sub="в среднем по библиотеке" isDark />
```
