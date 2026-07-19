import React, { useState } from 'react';
import { ModeCard } from 'musix-ui';

const obDark = {
  '--ob-card-bg': 'linear-gradient(180deg,#1d1d23,#131318)',
  '--ob-card-edge': 'rgba(255,255,255,.05)',
  padding: 26, borderRadius: 16,
  background: 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)',
  color: '#eeeef3', fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'grid', gridTemplateColumns: 'repeat(2, minmax(220px, 1fr))', gap: 12,
} as React.CSSProperties;

export const ModePickDark = () => {
  const [mode, setMode] = useState('server');
  return (
    <div style={obDark}>
      <ModeCard sel={mode === 'server'} onClick={() => setMode('server')}
        title="Сервер" body="Участники загружают музыку на этот сервер и слушают общую библиотеку."
        note="Режим фиксируется при создании владельца" />
      <ModeCard sel={mode === 'sharing'} onClick={() => setMode('sharing')}
        title="Шаринг" body="Каждый участник индексирует свою локальную папку с музыкой." />
    </div>
  );
};
