---
category: Controls
---

# CrossfadeText

Кроссфейд подписи при смене `text`: уходящая строка уплывает вверх, входящая
поднимается снизу (классы `.xfade*` из styles.css; абсолютное наложение — никогда
жёсткой замены). Для статусных строк, названий текущего шага агента и т.п.

```tsx
import { CrossfadeText } from 'musix-ui';

<CrossfadeText text={currentStageLabel} />
```
