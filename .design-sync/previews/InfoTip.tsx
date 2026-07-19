import React from 'react';
import { InfoTip } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: '46px 24px 22px', background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 26, alignItems: 'center', fontSize: 13.5,
};

// Карточка-подсказка появляется по hover/фокусу.
export const InLabelsDark = () => (
  <div style={dark}>
    <span style={{ display: 'inline-flex', gap: 7, alignItems: 'center' }}>
      Дослушивание
      <InfoTip isDark text="Учитываются только прослушивания длиннее 30 секунд" />
    </span>
    <span style={{ display: 'inline-flex', gap: 7, alignItems: 'center', color: 'rgba(255,255,255,.6)' }}>
      Индекс новизны
      <InfoTip isDark size={13} text="Доля треков, добавленных за последние 90 дней" />
    </span>
  </div>
);

export const Light = () => (
  <div style={{ ...dark, background: '#f2f1f6', color: '#161620' }}>
    <span style={{ display: 'inline-flex', gap: 7, alignItems: 'center' }}>
      Серия дней
      <InfoTip isDark={false} text="Дни подряд хотя бы с одним прослушиванием" />
    </span>
  </div>
);
