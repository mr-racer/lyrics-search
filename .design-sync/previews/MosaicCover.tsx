import React from 'react';
import { MosaicCover } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14,
  display: 'flex', gap: 18, alignItems: 'flex-end', flexWrap: 'wrap',
};

// Без coverPaths каждая ячейка — градиент-заглушка; в приложении сюда
// приходят обложки первых четырёх треков плейлиста.
export const TileCounts = () => (
  <div style={dark}>
    <MosaicCover trackIds={[]} coverPaths={[]} size={96} />
    <MosaicCover trackIds={[1]} coverPaths={[null]} size={96} />
    <MosaicCover trackIds={[1, 2]} coverPaths={[null, null]} size={96} />
    <MosaicCover trackIds={[1, 2, 3]} coverPaths={[null, null, null]} size={96} />
    <MosaicCover trackIds={[1, 2, 3, 4]} coverPaths={[null, null, null, null]} size={128} />
  </div>
);
