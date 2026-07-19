import React from 'react';
import { OBStageBar, useColors } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: '24px 26px', background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif", minWidth: 380,
};

export const IndexingStagesDark = () => {
  const c = useColors(true);
  return (
    <div style={dark}>
      <OBStageBar c={c} label="СКАН ФАЙЛОВ" state="done" pct={100} />
      <OBStageBar c={c} label="ТЕКСТЫ ПЕСЕН" state="done" pct={100} />
      <OBStageBar c={c} label="ЭМБЕДДИНГИ" state="running" pct={62} count="812/1310" eta="~4 мин" />
      <OBStageBar c={c} label="АНАЛИЗ ЗВУКА" state="running" indeterminate />
      <OBStageBar c={c} label="ПОХОЖЕСТЬ" state="pending" pct={0} />
      <OBStageBar c={c} label="ОБЛОЖКИ" state="failed" pct={40} />
    </div>
  );
};

export const Light = () => {
  const c = useColors(false);
  return (
    <div style={{ ...dark, background: '#f2f1f6', color: '#161620' }}>
      <OBStageBar c={c} label="СКАН ФАЙЛОВ" state="done" pct={100} />
      <OBStageBar c={c} label="ЭМБЕДДИНГИ" state="running" pct={38} count="205/540" />
    </div>
  );
};
