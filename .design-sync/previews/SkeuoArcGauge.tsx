import React from 'react';
import { SkeuoArcGauge } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 26, alignItems: 'flex-start', flexWrap: 'wrap',
};

export const GaugesDark = () => (
  <div style={dark}>
    <SkeuoArcGauge value={0.78} hue={145} label="Дослушивание" sub="в среднем по библиотеке" isDark />
    <SkeuoArcGauge value={0.34} hue={30} label="Скипы" sub="за последний месяц" isDark />
  </div>
);

export const Light = () => (
  <div style={{ ...dark, background: '#f2f1f6', color: '#161620' }}>
    <SkeuoArcGauge value={0.61} hue={275} label="Новизна" sub="доля свежих треков" isDark={false} />
  </div>
);
