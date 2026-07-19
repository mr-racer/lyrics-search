import React, { useState } from 'react';
import { SectionHeader, Segmented } from 'musix-ui';

const page: React.CSSProperties = {
  background: '#0d0d10', borderRadius: 14, overflow: 'hidden', minWidth: 480,
  color: '#eeeef3', fontFamily: "'Geist','Inter',system-ui,sans-serif",
};

export const WithTabsDark = () => {
  const [tab, setTab] = useState('albums');
  return (
    <div style={page}>
      <SectionHeader isDark right={
        <Segmented isDark value={tab} onChange={setTab} options={[
          { value: 'albums', label: 'Альбомы' },
          { value: 'recent', label: 'Недавние' },
          { value: 'stats', label: 'Статистика' },
        ]} />
      } />
      <div style={{ padding: '18px 36px 26px', fontSize: 13, color: 'rgba(255,255,255,.45)' }}>
        …контент секции под шапкой…
      </div>
    </div>
  );
};

export const WithChildrenLight = () => (
  <div style={{ ...page, background: '#f2f1f6', color: '#161620' }}>
    <SectionHeader isDark={false}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
        <span style={{ fontWeight: 700, fontSize: 15 }}>Импорт из Яндекс Музыки</span>
        <span style={{ fontSize: 12, color: 'rgba(10,10,18,.5)' }}>142 трека в очереди</span>
      </div>
    </SectionHeader>
    <div style={{ padding: '18px 36px 26px', fontSize: 13, color: 'rgba(10,10,18,.5)' }}>
      …контент секции…
    </div>
  </div>
);
