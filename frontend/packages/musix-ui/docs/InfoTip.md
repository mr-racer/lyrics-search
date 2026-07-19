---
category: Controls
---

# InfoTip

Маленький значок «i» с всплывающей карточкой (250px) по hover/фокусу. Полностью
инлайновые стили, тема через `isDark`. Легче HintBadge: без порталов и мобильного
bottom sheet — для плотных мест (шапки панелей, строки настроек).

```tsx
import { InfoTip } from 'musix-ui';

<InfoTip isDark text="Учитываются только прослушивания длиннее 30 секунд" />
```
