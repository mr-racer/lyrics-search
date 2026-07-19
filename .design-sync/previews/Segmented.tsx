import React, { useState } from 'react';
import { Segmented } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 18, alignItems: 'center', flexWrap: 'wrap',
};
const light: React.CSSProperties = { ...dark, background: '#f2f1f6', color: '#161620' };

export const LibraryTabsDark = () => {
  const [tab, setTab] = useState('albums');
  return (
    <div style={dark}>
      <Segmented isDark value={tab} onChange={setTab} options={[
        { value: 'albums', label: 'Альбомы' },
        { value: 'recent', label: 'Недавние' },
        { value: 'playlists', label: 'Плейлисты' },
        { value: 'stats', label: 'Статистика' },
      ]} />
    </div>
  );
};

export const SmallSortLight = () => {
  const [sort, setSort] = useState('recent');
  return (
    <div style={light}>
      <Segmented isDark={false} size="sm" value={sort} onChange={setSort} options={[
        { value: 'recent', label: 'Свежее' },
        { value: 'alpha', label: 'А–Я' },
        { value: 'plays', label: 'По прослушиваниям' },
      ]} />
    </div>
  );
};
