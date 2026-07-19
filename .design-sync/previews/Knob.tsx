import React, { useState } from 'react';
import { Knob } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 22, alignItems: 'center',
};

export const ThemeKnobDark = () => {
  const [on, setOn] = useState(false);
  return (
    <div style={dark}>
      <Knob size={36} isDark angle={on ? 55 : -55} glow={on} label="Тема" onClick={() => setOn(!on)} />
      <Knob size={44} isDark angle={-20} label="Фильтр" onClick={() => {}} />
      <Knob size={56} isDark angle={30} glow label="Режим" onClick={() => {}} />
    </div>
  );
};

export const Light = () => (
  <div style={{ ...dark, background: '#f2f1f6', color: '#161620' }}>
    <Knob size={36} isDark={false} angle={55} glow label="Тема" onClick={() => {}} />
    <Knob size={44} isDark={false} angle={-45} label="Фильтр" onClick={() => {}} />
  </div>
);
