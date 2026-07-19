import React from 'react';
import { GlassCard } from 'musix-ui';

// .ob-glass читает --ob-glass-* переменные — в приложении их ставит корень
// онбординга; здесь зеркалим те же значения на обёртке.
const obDark = {
  '--ob-glass-bg': 'rgba(255,255,255,.055)',
  '--ob-glass-sheen': 'rgba(255,255,255,.12)',
  '--ob-glass-edge': 'rgba(255,255,255,.18)',
  padding: 28, borderRadius: 16,
  background: 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)',
  color: '#eeeef3', fontFamily: "'Geist','Inter',system-ui,sans-serif",
} as React.CSSProperties;

export const WelcomeDark = () => (
  <div style={obDark}>
    <GlassCard style={{ padding: '26px 30px', maxWidth: 420 }}>
      <div style={{ fontSize: 19, fontWeight: 700, marginBottom: 8 }}>Добро пожаловать в MusiX</div>
      <div style={{ fontSize: 13.5, lineHeight: 1.5, opacity: .72 }}>
        Локальный плеер с семантическим поиском: тексты, звучание и вкус — всё на вашей машине.
      </div>
    </GlassCard>
  </div>
);

export const LightSurface = () => (
  <div style={{
    ...obDark,
    '--ob-glass-bg': 'rgba(255,255,255,.62)',
    '--ob-glass-sheen': 'rgba(255,255,255,.9)',
    '--ob-glass-edge': 'rgba(0,0,0,.10)',
    background: 'linear-gradient(180deg,#f5f4fa,#e6e4f0)',
    color: '#161620',
  } as React.CSSProperties}>
    <GlassCard style={{ padding: '22px 26px', maxWidth: 420 }}>
      <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 6 }}>Настройка завершена</div>
      <div style={{ fontSize: 13, lineHeight: 1.5, opacity: .7 }}>Библиотека проиндексирована — можно слушать.</div>
    </GlassCard>
  </div>
);
