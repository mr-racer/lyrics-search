import React from 'react';
import { Skel } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 22, background: '#0d0d10', borderRadius: 14,
  display: 'flex', flexDirection: 'column', gap: 12, minWidth: 320,
};

export const TrackRowSkeleton = () => (
  <div style={dark}>
    {[0, 1, 2].map(i => (
      <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <Skel w={44} h={44} r={10} isDark />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 7 }}>
          <Skel w="62%" h={13} isDark />
          <Skel w="38%" h={11} isDark />
        </div>
        <Skel w={54} h={22} r={12} isDark />
      </div>
    ))}
  </div>
);

export const CardSkeleton = () => (
  <div style={dark}>
    <Skel w="100%" h={120} r={14} isDark />
    <Skel w="70%" h={14} isDark />
    <Skel w="45%" h={12} isDark />
  </div>
);
