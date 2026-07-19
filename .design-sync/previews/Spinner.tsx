import React from 'react';
import { Spinner } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 22, alignItems: 'center', flexWrap: 'wrap',
};

export const Sizes = () => (
  <div style={dark}>
    <Spinner size={14} />
    <Spinner size={20} />
    <Spinner size={28} />
    <span style={{ display: 'inline-flex', gap: 10, alignItems: 'center', fontSize: 13, color: 'rgba(255,255,255,.6)' }}>
      <Spinner size={16} /> Ищу похожие треки…
    </span>
  </div>
);

export const Accents = () => (
  <div style={dark}>
    <Spinner size={24} color="oklch(60% 0.18 270)" />
    <Spinner size={24} color="oklch(72% 0.13 75)" />
    <Spinner size={24} color="oklch(63% 0.17 142)" />
    <Spinner size={24} color="oklch(58% 0.21 25)" />
  </div>
);
