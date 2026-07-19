---
category: Layout
---

# SectionHeader

Шапка секции: тонкая градиентная полоса с нижней границей. Хост для `right`
(табы/фильтры, прижимаются вправо) и `children`; без обоих рендерит `null`
(коллапсирует полностью).

```tsx
import { SectionHeader, Segmented } from 'musix-ui';

<SectionHeader isDark right={
  <Segmented value={tab} onChange={setTab} isDark
    options={[{ value: 'albums', label: 'Альбомы' }, { value: 'recent', label: 'Недавние' }]} />
} />
```
