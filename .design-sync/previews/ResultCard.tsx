import React from 'react';
import { ResultCard, useColors } from 'musix-ui';

const wrapD: React.CSSProperties = {
  padding: 16, background: '#0d0d10', borderRadius: 14,
  fontFamily: "'Geist','Inter',system-ui,sans-serif", maxWidth: 560,
};
const wrapL: React.CSSProperties = { ...wrapD, background: '#f2f1f6' };

const hits = {
  lyrics: { score: 0.87, matched_on: 'lyrics', track: { title: 'Группа крови', artist: 'Кино', year: 1988, genre: 'post-punk' } },
  audio: { score: 0.74, matched_on: 'audio', track: { title: 'Искала', artist: 'Земфира', year: 2000, genre: 'rock' } },
  hybrid: { score: 0.91, matched_on: 'hybrid', track: { title: 'Полковнику никто не пишет', artist: 'Би-2', year: 2000, genre: 'alt-rock' } },
} as const;

export const MatchModesDark = () => {
  const c = useColors(true);
  return (
    <div style={wrapD}>
      <ResultCard isDark c={c} hit={hits.lyrics} onClick={() => {}} onPlay={() => {}} />
      <ResultCard isDark c={c} hit={hits.audio} onClick={() => {}} onPlay={() => {}} />
      <ResultCard isDark c={c} hit={hits.hybrid} onClick={() => {}} />
    </div>
  );
};

export const CompactLight = () => {
  const c = useColors(false);
  return (
    <div style={wrapL}>
      <ResultCard isDark={false} c={c} compact hit={hits.lyrics} onClick={() => {}} onPlay={() => {}} />
      <ResultCard isDark={false} c={c} compact hit={hits.audio} onClick={() => {}} />
    </div>
  );
};
