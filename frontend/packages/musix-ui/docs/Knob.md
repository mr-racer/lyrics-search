---
category: Controls
---

# Knob

Круглая скеуоморфная ручка-крутилка с меткой-риской. `angle` поворачивает риску
(−90..90), `glow` подсвечивает её янтарным. В приложении — переключатель темы
(angle −55 в тёмной, +55 со свечением в светлой).

```tsx
import { Knob } from 'musix-ui';

<Knob size={36} isDark angle={-55} label="Светлая тема" onClick={toggleTheme} />
<Knob size={36} isDark={false} angle={55} glow label="Тёмная тема" onClick={toggleTheme} />
```
