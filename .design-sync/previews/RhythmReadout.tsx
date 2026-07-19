import React from 'react';
import { RhythmReadout } from 'musix-ui';

const wrapD: React.CSSProperties = {
  padding: 24, borderRadius: 14, fontFamily: "'Geist','Inter',system-ui,sans-serif",
  background: 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 100%)',
  display: 'flex', gap: 12, flexWrap: 'wrap',
};
const wrapL: React.CSSProperties = { ...wrapD, background: 'linear-gradient(180deg,#f5f4fa,#e9e8f0)' };

export const StreakRowDark = () => (
  <div style={wrapD}>
    <RhythmReadout icon="🔥" value="12 дней" label="Серия" foot="лучшая — 21 день" hue={30} isDark />
    <RhythmReadout icon="🎧" value="Пятница" label="Самый активный день" hue={275} isDark />
    <RhythmReadout icon="🌙" value="23:00–01:00" label="Пик прослушиваний" foot="ночной слушатель" hue={215} isDark grow />
  </div>
);

export const Light = () => (
  <div style={wrapL}>
    <RhythmReadout icon="☀️" value="4 ч 12 мин" label="Сегодня" hue={145} isDark={false} />
    <RhythmReadout icon="📀" value="Мракобесие и джаз" label="Альбом недели" foot="Агата Кристи · 14 прослушиваний" hue={275} isDark={false} grow />
  </div>
);
