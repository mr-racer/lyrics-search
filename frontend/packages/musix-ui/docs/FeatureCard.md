---
category: Layout
---

# FeatureCard

Карточка фичи на экранах онбординга: крупная эмодзи-иконка, заголовок, описание.
`premium` — золотая рамка (`.ob-feat-premium`), иначе hover-подъём (`.ob-feat-hover`).
Классы `.ob-feat*` идут из styles.css.

```tsx
import { FeatureCard } from 'musix-ui';

<FeatureCard icon="🔍" title="Поиск по смыслу" body="Ищите песни по настроению и теме, а не только по названию." />
<FeatureCard icon="✨" premium title="AI-подборки" body="Плейлисты под запрос «дождливый вечер»." />
```
