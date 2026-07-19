import React from 'react';
import { LazyCover } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14,
  display: 'flex', gap: 14, alignItems: 'center',
};

// Без url рендерится фирменный градиент-заглушка; с битым url — тоже
// (ошибка откатывается на градиент, а не на «битую» иконку).
export const Fallbacks = () => (
  <div style={dark}>
    <LazyCover style={{ width: 46, height: 46, borderRadius: 10 }} />
    <LazyCover style={{ width: 64, height: 64, borderRadius: 12 }}
      fallback="linear-gradient(135deg, rgba(77,214,194,.4), rgba(77,125,255,.25))" />
    <LazyCover style={{ width: 96, height: 96, borderRadius: 14 }} />
  </div>
);
