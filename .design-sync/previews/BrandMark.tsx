import React from 'react';
import { BrandMark } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 24, alignItems: 'center',
};

export const WordmarkDark = () => (
  <div style={dark}>
    <span style={{ display: 'inline-flex', gap: 10, alignItems: 'center' }}>
      <BrandMark size={34} isDark />
      <span style={{ fontWeight: 700, fontSize: 18, letterSpacing: '-0.01em' }}>MusiX</span>
    </span>
    <BrandMark size={48} isDark />
    <BrandMark size={24} isDark />
  </div>
);

export const Light = () => (
  <div style={{ ...dark, background: '#f2f1f6', color: '#161620' }}>
    <span style={{ display: 'inline-flex', gap: 10, alignItems: 'center' }}>
      <BrandMark size={34} isDark={false} />
      <span style={{ fontWeight: 700, fontSize: 18 }}>MusiX</span>
    </span>
  </div>
);
