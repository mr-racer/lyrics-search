import React from 'react';
import { CompletionRing } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap',
};

export const RangeDark = () => (
  <div style={dark}>
    <CompletionRing pct={0.96} hue={145} isDark />
    <CompletionRing pct={0.71} hue={145} isDark />
    <CompletionRing pct={0.42} hue={75} isDark />
    <CompletionRing pct={0.18} hue={30} isDark />
    <CompletionRing pct={0.84} hue={275} size={56} isDark />
  </div>
);

export const InRow = () => (
  <div style={{ ...dark, gap: 12 }}>
    <CompletionRing pct={0.92} hue={145} isDark size={38} />
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <span style={{ fontSize: 13.5, fontWeight: 600 }}>Кукушка</span>
      <span style={{ fontSize: 12, color: 'rgba(255,255,255,.5)' }}>Кино · дослушивают 92%</span>
    </div>
  </div>
);
