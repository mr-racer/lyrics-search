---
category: Controls
---

# Segmented

Сегментированный контрол во «врезанном» жёлобе (`ske-inset-*`): активный сегмент —
рельефная кнопка (`ske-btn-*`), неактивные — плоский текст. Размеры: `md` (по умолчанию) и `sm`.

```tsx
import { Segmented } from 'musix-ui';

<Segmented
  value={tab}
  onChange={setTab}
  isDark={isDark}
  options={[
    { value: 'albums',  label: 'Альбомы' },
    { value: 'recent',  label: 'Недавние' },
    { value: 'stats',   label: 'Статистика' },
  ]}
/>
```

Используется для табов внутри секции и переключателей сортировки.
