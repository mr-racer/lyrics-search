---
category: Layout
---

# Empty

Минимальная курсивная заглушка пустого состояния: «Нет данных» / “No data” по `lang`.
Для панелей статистики и списков, где пустота — норма и не заслуживает иллюстрации.

```tsx
import { Empty } from 'musix-ui';

{rows.length ? <Rows /> : <Empty lang="ru" />}
```
