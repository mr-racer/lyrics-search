import React from 'react';
import { FeatureCard } from 'musix-ui';

const obDark = {
  '--ob-glass-sheen': 'rgba(255,255,255,.12)',
  '--ob-card-bg': 'linear-gradient(180deg,#1d1d23,#131318)',
  '--ob-card-edge': 'rgba(255,255,255,.05)',
  padding: 26, borderRadius: 16,
  background: 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)',
  color: '#eeeef3', fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'grid', gridTemplateColumns: 'repeat(2, minmax(200px, 1fr))', gap: 14,
} as React.CSSProperties;

export const FeaturesGridDark = () => (
  <div style={obDark}>
    <FeatureCard icon="🔍" title="Поиск по смыслу" body="Ищите песни по настроению и теме, а не только по названию." />
    <FeatureCard icon="🎧" title="Бесконечный поток" body="Персональный стрим подстраивается под реакции на лету." />
    <FeatureCard icon="✨" premium title="AI-подборки" body="Плейлисты под запрос «дождливый вечер у окна»." />
    <FeatureCard icon="🗺️" premium title="Карта вкуса" body="Острова вашей библиотеки и звуковой портрет." />
  </div>
);
