import React, { useState } from 'react';
import { ToggleSwitch } from 'musix-ui';

const row: React.CSSProperties = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 };
const dark: React.CSSProperties = {
  padding: 22, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', flexDirection: 'column', gap: 14, minWidth: 300, fontSize: 13.5,
};

export const SettingsDark = () => {
  const [a, setA] = useState(true);
  const [b, setB] = useState(false);
  return (
    <div style={dark}>
      <div style={row}><span>Автовоспроизведение похожих</span><ToggleSwitch checked={a} onChange={setA} isDark /></div>
      <div style={row}><span>Скеуоморфные анимации</span><ToggleSwitch checked={b} onChange={setB} isDark /></div>
    </div>
  );
};

export const Light = () => {
  const [a, setA] = useState(true);
  return (
    <div style={{ ...dark, background: '#f2f1f6', color: '#161620' }}>
      <div style={row}><span>Уведомления о новых фактах</span><ToggleSwitch checked={a} onChange={setA} isDark={false} /></div>
      <div style={row}><span>Тёмная тема по расписанию</span><ToggleSwitch checked={!a} onChange={() => setA(!a)} isDark={false} /></div>
    </div>
  );
};
