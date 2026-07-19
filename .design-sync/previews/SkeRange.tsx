import React, { useState } from 'react';
import { SkeRange } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: '26px 24px', background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', flexDirection: 'column', gap: 20, minWidth: 320,
};
// Светлая тема в приложении включается body[data-theme="light"];
// в карточке зеркалим её переменные локально через style.
const lightVars = {
  '--rng-groove': 'oklch(82% 0.012 280)',
  '--rng-groove-hi': 'rgba(255,255,255,0.7)',
  '--rng-knob-top': 'oklch(99% 0.004 280)',
  '--rng-knob-bot': 'oklch(88% 0.012 280)',
  '--rng-knob-ring': 'rgba(40,30,60,0.22)',
} as React.CSSProperties;

export const VolumeDark = () => {
  const [v, setV] = useState(64);
  return (
    <div style={dark}>
      <SkeRange value={v} onChange={setV} ariaLabel="Громкость" />
      <SkeRange value={30} onChange={() => {}} accent="oklch(78% 0.13 75)" ariaLabel="Тембр" />
      <SkeRange value={80} onChange={() => {}} accent="oklch(70% 0.13 165)" ariaLabel="Пространство" />
    </div>
  );
};

export const BipolarAndStates = () => {
  const [bal, setBal] = useState(2);
  return (
    <div style={dark}>
      <SkeRange value={bal} min={-5} max={5} bipolar animated onChange={setBal} ariaLabel="Баланс" />
      <SkeRange value={45} disabled onChange={() => {}} ariaLabel="Недоступно" />
    </div>
  );
};

export const Light = () => {
  const [v, setV] = useState(52);
  return (
    <div style={{ ...dark, background: '#f2f1f6', color: '#161620' }}>
      <SkeRange value={v} onChange={setV} style={lightVars} ariaLabel="Громкость" />
      <SkeRange value={25} onChange={() => {}} style={lightVars} accent="oklch(58% 0.15 60)" ariaLabel="Тембр" />
    </div>
  );
};
