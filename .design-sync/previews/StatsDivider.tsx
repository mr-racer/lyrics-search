import React from 'react';
import { StatsDivider } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: '22px 24px', background: '#0d0d10', borderRadius: 14,
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', flexDirection: 'column', gap: 22, minWidth: 420,
};

export const HuesDark = () => (
  <div style={dark}>
    <StatsDivider label="Прослушивания" hue={145} />
    <StatsDivider label="Коллекция" hue={275} />
    <StatsDivider label="Ритм недели" hue={30} />
  </div>
);
