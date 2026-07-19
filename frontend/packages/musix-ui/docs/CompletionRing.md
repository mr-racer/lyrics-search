---
category: Data
---

# CompletionRing

Компактное кольцо завершённости (0..1) с числом процентов в центре и мягким
свечением дуги. Для строк списков («любимые» — доля дослушиваний).

```tsx
import { CompletionRing } from 'musix-ui';

<CompletionRing pct={0.92} hue={145} isDark />   // дослушивают почти всегда
<CompletionRing pct={0.34} hue={30} isDark />    // часто скипают
```
