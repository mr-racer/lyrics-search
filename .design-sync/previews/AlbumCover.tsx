import React from 'react';
import { AlbumCover } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 18, alignItems: 'flex-end', flexWrap: 'wrap',
};

// Без coverPath обложка — детерминированный градиент с инициалами
// (в живом приложении сюда приходит арт с сервера).
export const GradientFallbacks = () => (
  <div style={dark}>
    <AlbumCover title="Группа крови" artist="Кино" size={44} isDark />
    <AlbumCover title="Искала" artist="Земфира" size={64} isDark />
    <AlbumCover title="Мракобесие и джаз" artist="Агата Кристи" size={110} isDark />
    <AlbumCover title="Полковнику никто не пишет" artist="Би-2" size={160} isDark />
  </div>
);

export const FluidGrid = () => (
  <div style={{ ...dark, display: 'grid', gridTemplateColumns: 'repeat(4, 96px)', gap: 12, alignItems: 'stretch' }}>
    {[['Лето', 'Кино'], ['Прогулки', 'Наутилус'], ['Танцы', 'Мумий Тролль'], ['Вечер', 'Сплин']].map(([t, a]) => (
      <div key={t} style={{ aspectRatio: '1 / 1' }}>
        <AlbumCover fluid title={t} artist={a} isDark radius={12} />
      </div>
    ))}
  </div>
);
