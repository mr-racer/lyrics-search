---
category: Controls
---

# ToggleSwitch

Двухпозиционный переключатель-«рокер»: две половинки, активная подсвечена фирменным
фиолетовым градиентом с внутренним свечением. Ширина фиксированная (44×22).

```tsx
import { ToggleSwitch } from 'musix-ui';

const [on, setOn] = useState(true);
<ToggleSwitch checked={on} onChange={setOn} isDark={isDark} />
```

Используется в настройках и фильтрах; подпись размещайте снаружи (слева/справа от контрола).
