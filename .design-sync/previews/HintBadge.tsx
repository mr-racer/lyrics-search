import React from 'react';
import { HintBadge } from 'musix-ui';

const dark: React.CSSProperties = {
  padding: '48px 24px 24px', background: '#0d0d10', borderRadius: 14, color: '#eeeef3',
  fontFamily: "'Geist','Inter',system-ui,sans-serif",
  display: 'flex', gap: 26, alignItems: 'center', fontSize: 13.5,
};

// Тултип раскрывается по hover/фокусу — в статичной карточке виден сам значок;
// наведите курсор в живом превью.
export const InRowDark = () => (
  <div style={dark}>
    <span style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
      Похожесть звука
      <HintBadge size={18} label="Считается по CLAP-эмбеддингам аудио, не по жанровым тегам" />
    </span>
    <span style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
      Гибридный поиск
      <HintBadge size={16} placement="down" label="RRF-фьюжн плотного и bm25-разреженного индексов" />
    </span>
  </div>
);
