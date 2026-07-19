import React, { useEffect, useState } from 'react';
import { CrossfadeText } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: 24, background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif", fontSize: 14,
  display: 'flex', alignItems: 'center', gap: 10, minHeight: 64,
};

const STAGES = ['Думаю над запросом…', 'Ищу по текстам…', 'Сверяю звучание…', 'Собираю ответ…'];

// Живой цикл: подпись плавно сменяется каждые 1.6 с.
export const AgentStages = () => {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI(x => (x + 1) % STAGES.length), 1600);
    return () => clearInterval(t);
  }, []);
  return (
    <div style={dark}>
      <span style={{ opacity: .5, fontSize: 12, letterSpacing: '.12em' }}>АГЕНТ</span>
      <CrossfadeText text={STAGES[i]} />
    </div>
  );
};
