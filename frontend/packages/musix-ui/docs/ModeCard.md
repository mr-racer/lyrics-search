---
category: Layout
---

# ModeCard

Выбираемая карточка режима в мастере настройки. `sel` подсвечивает фиолетовой
рамкой-свечением; `note` — сноска в сиреневой плашке. Требует CSS-переменные
`--ob-card-bg` и `--ob-card-edge` на родителе.

```tsx
import { ModeCard } from 'musix-ui';

<div style={{ '--ob-card-bg': 'rgba(255,255,255,.05)', '--ob-card-edge': 'rgba(255,255,255,.12)' } as React.CSSProperties}>
  <ModeCard sel title="Сервер" body="Участники загружают музыку на этот сервер."
    note="Режим фиксируется при создании владельца" onClick={pickServer} />
  <ModeCard title="Шаринг" body="Каждый слушает свою локальную библиотеку." onClick={pickSharing} />
</div>
```
