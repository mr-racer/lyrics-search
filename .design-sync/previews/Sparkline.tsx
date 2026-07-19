import React from 'react';
import { Sparkline } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 26, alignItems: 'center', flexWrap: 'wrap',
};

// Даты строим от «сегодня», чтобы график всегда был живым.
const daysBack = (counts: number[]) => {
  const today = new Date(); today.setHours(0, 0, 0, 0);
  return counts.map((count, i) => {
    const d = new Date(today); d.setDate(today.getDate() - (counts.length - 1 - i));
    const iso = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    return { date: iso, count };
  });
};

export const ActivityDark = () => (
  <div style={dark}>
    <span style={{ display: 'inline-flex', gap: 10, alignItems: 'center', fontSize: 12, color: 'rgba(255,255,255,.55)' }}>
      Активная неделя
      <Sparkline hue={145} days={daysBack([2, 5, 3, 8, 12, 9, 14, 11, 16, 13, 18, 22, 19, 24])} />
    </span>
    <span style={{ display: 'inline-flex', gap: 10, alignItems: 'center', fontSize: 12, color: 'rgba(255,255,255,.55)' }}>
      Спад
      <Sparkline hue={30} days={daysBack([20, 16, 18, 12, 9, 11, 6, 8, 4, 5, 2, 3, 1, 0])} />
    </span>
    <span style={{ display: 'inline-flex', gap: 10, alignItems: 'center', fontSize: 12, color: 'rgba(255,255,255,.55)' }}>
      Крупнее
      <Sparkline hue={275} width={160} height={36} days={daysBack([4, 9, 6, 14, 10, 18, 12, 20, 15, 24, 17, 28, 21, 30])} />
    </span>
  </div>
);
