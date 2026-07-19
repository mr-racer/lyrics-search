import React from 'react';
import { Empty } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 22, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', flexDirection: 'column', gap: 10, minWidth: 260,
};

export const InPanel = () => (
  <div style={dark}>
    <div style={{ fontSize: 11, letterSpacing: '.18em', opacity: .5 }}>ЖАНРЫ ЗА НЕДЕЛЮ</div>
    <Empty lang="ru" />
  </div>
);

export const English = () => (
  <div style={dark}>
    <div style={{ fontSize: 11, letterSpacing: '.18em', opacity: .5 }}>RECENT SKIPS</div>
    <Empty lang="en" />
  </div>
);
