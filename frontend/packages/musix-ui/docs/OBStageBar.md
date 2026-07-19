---
category: Data
---

# OBStageBar

Строка прогресса этапа (онбординг/индексация): подпись слева, статус справа
(✓ done, ✗ failed, … running, · pending), полоса с процентом или indeterminate-шиммером
(`.ob-indet`). Принимает палитру `c` из `useColors(isDark)`; running-этап может
показывать счётчик `count` и `eta`.

```tsx
import { OBStageBar, useColors } from 'musix-ui';

const c = useColors(true);
<OBStageBar c={c} label="СКАН ФАЙЛОВ" state="done" pct={100} />
<OBStageBar c={c} label="ЭМБЕДДИНГИ" state="running" pct={62} count="812/1310" eta="~4 мин" />
<OBStageBar c={c} label="АНАЛИЗ ЗВУКА" state="pending" pct={0} />
```
