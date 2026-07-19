import React from 'react';
import { StepCheck } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 22, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', flexDirection: 'column', gap: 10, fontSize: 13.5,
};
const done: React.CSSProperties = { display: 'inline-flex', gap: 8, alignItems: 'center', color: 'oklch(63% 0.17 142)' };
const pending: React.CSSProperties = { display: 'inline-flex', gap: 8, alignItems: 'center', color: 'rgba(255,255,255,.4)' };

export const WizardSteps = () => (
  <div style={dark}>
    <span style={done}><StepCheck size={12} /> Файлы просканированы</span>
    <span style={done}><StepCheck size={12} /> Тексты найдены</span>
    <span style={pending}>· Эмбеддинги строятся…</span>
  </div>
);
