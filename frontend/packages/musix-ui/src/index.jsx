// musix-ui — презентационные примитивы MusiX (Studio Console).
// Код физически перенесён из frontend/src/main.jsx; приложение импортирует
// компоненты обратно отсюда. Стили: инлайн + классы из frontend/src/index.css
// (.ske-*, .brushed-*, .rec-div, .ob-*, .load-skel, .hint-badge, .xfade…).
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';

const API = (typeof window !== 'undefined' && window.location.protocol === 'file:')
  ? 'http://localhost:8000/api/v1'
  : '/api/v1';

// ─── Color tokens ─────────────────────────────────────────────────────────────
/** Палитра MusiX для выбранной темы. Возвращает объект токенов (bg, surface, text, accent, amber, green, red…) — компоненты берут цвета отсюда, а не из CSS-переменных. */
export function useColors(isDark) {
  return useMemo(() => ({
    bg:           isDark ? '#0d0d10'    : '#f2f1f6',
    bgDeep:       isDark ? '#08080b'    : '#e8e7ed',
    surface:      isDark ? '#17171b'    : '#ffffff',
    surface2:     isDark ? '#1e1e24'    : '#ebebf0',
    surface3:     isDark ? '#26262d'    : '#e4e3e9',
    border:       isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.11)',
    borderStrong: isDark ? 'rgba(255,255,255,0.13)' : 'rgba(0,0,0,0.20)',
    text:         isDark ? '#eeeef3'    : '#161620',
    // Light values are deliberately denser than a naive mirror of the dark
    // alphas: black-on-#f2f1f6 loses contrast faster than white-on-#0d0d10,
    // and these two tokens carry most 10-12px labels (WCAG AA ≥ 4.5:1).
    textMuted:    isDark ? 'rgba(255,255,255,0.5)' : 'rgba(10,10,18,0.60)',
    textSubtle:   isDark ? 'rgba(255,255,255,0.28)' : 'rgba(10,10,18,0.46)',
    accent:       'oklch(60% 0.18 270)',
    accentBg:     isDark ? 'oklch(60% 0.18 270 / 0.15)' : 'oklch(60% 0.18 270 / 0.10)',
    accentLight:  'oklch(70% 0.16 270)',
    amber:        isDark ? 'oklch(72% 0.13 75)' : 'oklch(58% 0.15 60)',
    amberGlow:    'rgba(212,165,90,0.35)',
    inputBg:      isDark ? '#10101a'    : '#ffffff',
    sidebarBg:    isDark ? 'linear-gradient(180deg,#131318 0%,#0c0c10 100%)' : 'linear-gradient(180deg,#f0eff5 0%,#e7e6ec 100%)',
    chatPanelBg:  isDark ? '#0f0f13'    : '#f8f8fc',
    userBubble:   'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 305))',
    aiBubble:     isDark ? '#1e1e26'    : '#ebebf2',
    green:        'oklch(63% 0.17 142)',
    greenBg:      isDark ? 'oklch(63% 0.17 142 / 0.13)' : 'oklch(63% 0.17 142 / 0.10)',
    red:          'oklch(58% 0.21 25)',
    redBg:        isDark ? 'oklch(58% 0.21 25 / 0.14)' : 'oklch(58% 0.21 25 / 0.09)',
  }), [isDark]);
}
/** Имя скеуоморфного CSS-класса поверхности: ske('panel'|'btn'|'inset'|'display', isDark) → 'ske-panel-d' и т.п. */
export const ske = (kind, isDark) => `ske-${kind}-${isDark ? 'd' : 'l'}`;
/** Имя класса «шлифованный металл» для верхних дек: brushed(isDark) → 'brushed-d'/'brushed-l'. */
export const brushed = (isDark) => `brushed-${isDark ? 'd' : 'l'}`;

/** Отслеживает OS-настройку prefers-reduced-motion; анимации ветвятся по ней. */
export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false);
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const on = () => setReduced(mq.matches);
    mq.addEventListener?.('change', on);
    return () => mq.removeEventListener?.('change', on);
  }, []);
  return reduced;
}

/** true на «мобильной» ширине (≤768px), реагирует на изменение живьём. */
export function useIsMobile() {
  const MQ = '(max-width: 768px)';
  const [mobile, setMobile] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia(MQ).matches);
  useEffect(() => {
    const mql = window.matchMedia(MQ);
    const onChange = e => setMobile(e.matches);
    mql.addEventListener('change', onChange);
    setMobile(mql.matches);  // sync in case the breakpoint flipped before mount
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return mobile;
}

// ─── Cover thumbnails ─────────────────────────────────────────────────────────
// The covers endpoint generates and caches downscaled variants on ?w=320.
// Grids, rows and mosaics use them; the player (hero art, ambient wash, color
// sampling) keeps the full-size original — those three must share one URL so
// the browser cache is hit once and canvas sampling reuses the same entry.
/** URL миниатюры обложки (?w=320) для сеток и списков; полноразмер остаётся у плеера. */
export function thumbCoverUrl(url, w = 320) {
  if (!url || typeof url !== 'string') return url;
  if (url.startsWith('blob:') || url.startsWith('data:')) return url;
  if (/[?&]w=\d/.test(url)) return url;              // already a thumbnail URL
  return url.includes('?') ? `${url}&w=${w}` : `${url}?w=${w}`;
}

// ─── Spinner ──────────────────────────────────────────────────────────────────
/** Кольцевой индикатор загрузки. size — px, color — акцент (по умолчанию фирменный фиолетовый). Требует @keyframes spin из index.css. */
export function Spinner({ size = 16, color }) {
  return (
    <div style={{
      width: size, height: size, flexShrink: 0,
      border: `2px solid ${color ? color + '30' : 'rgba(255,255,255,0.12)'}`,
      borderTopColor: color || 'oklch(60% 0.18 270)',
      borderRadius: '50%', animation: 'spin 0.7s linear infinite',
    }} />
  );
}

// ─── SkeRange ───────────────────────────────────────────────────────────────
// Skeuomorphic slider: a carved groove + a raised knob. The native <input>
// rides transparently on top so drag / track-click / keyboard / a11y all work
// for free, while the visible fill + knob mirror its value. Knob & fill are
// inset by the knob radius so the knob never spills past the rail.
//   • accent   — per-instance fill/glow colour (purple, gold, …)
//   • animated — knob & fill glide between stops (use for fixed-value sliders;
//                leave off for continuous drags so the knob tracks the finger)
//   • bipolar  — fill grows out from the centre (zero) instead of the left
const SKE_RNG_KNOB = 18; // px — keep in sync with --rng-knob
/** Скеуоморфный слайдер: вырезанный жёлоб + рельефная ручка; нативный input сверху даёт drag/клавиатуру/a11y. accent задаёт цвет заливки/свечения, bipolar растит заливку от нуля в центре. */
export function SkeRange({ value, min = 0, max = 100, step = 1, onChange,
                    accent = 'oklch(62% 0.2 275)', animated = false,
                    bipolar = false, disabled = false, ariaLabel, style }) {
  const range = (max - min) || 1;
  const clamp = p => Math.max(0, Math.min(100, p));
  const pct = clamp(((value - min) / range) * 100);
  // Centre position of the knob at fraction p (inset by the knob radius).
  const center = p => `calc(${SKE_RNG_KNOB / 2}px + (100% - ${SKE_RNG_KNOB}px) * ${p / 100})`;

  let fillStyle;
  if (bipolar) {
    const zero = clamp(((0 - min) / range) * 100);
    const lo = Math.min(pct, zero);
    const hi = Math.max(pct, zero);
    fillStyle = { left: center(lo), width: `calc((100% - ${SKE_RNG_KNOB}px) * ${(hi - lo) / 100})` };
  } else {
    fillStyle = { left: 0, width: center(pct) };
  }

  const cls = 'ske-rng'
    + (animated ? ' is-animated' : '')
    + (disabled ? ' is-disabled' : '');

  return (
    <div className={cls} style={{ '--rng-accent': accent, ...style }}>
      <div className="ske-rng-track" />
      <div className="ske-rng-fill" style={fillStyle} />
      <div className="ske-rng-knob" style={{ left: center(pct) }} />
      <input type="range" className="ske-rng-input"
        min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={e => onChange(Number(e.target.value))}
        aria-label={ariaLabel} />
    </div>
  );
}

// ─── AlbumCover ───────────────────────────────────────────────────────────────
/** Обложка альбома: картинка с сервера (ленивая загрузка, серверная миниатюра) или детерминированный градиент с инициалами, если арта нет. fluid растягивает на контейнер. */
export function AlbumCover({ title='', artist='', size=44, isDark, coverPath, radius, fluid, eager }) {
  const hue = ((title.charCodeAt(0)||65)*37 + (artist.charCodeAt(0)||65)*17) % 360;
  const br = radius ?? (typeof size === 'number' ? (size > 100 ? 14 : size > 60 ? 10 : 8) : 0);
  const boxStyle = fluid
    ? { width:'100%', height:'100%', borderRadius:`${br}px`, overflow:'hidden',
        boxShadow: isDark
          ? 'inset 0 1px 0 rgba(255,255,255,0.14), inset 0 -1px 0 rgba(0,0,0,0.45), inset 0 0 0 1px rgba(0,0,0,0.5), 0 3px 8px rgba(0,0,0,0.55)'
          : 'inset 0 1px 0 rgba(255,255,255,0.85), inset 0 -1px 0 rgba(0,0,0,0.1), inset 0 0 0 1px rgba(0,0,0,0.08), 0 3px 7px rgba(40,30,60,0.13)' }
    : { width: size, height: size, borderRadius:`${br}px`, flexShrink:0, overflow:'hidden',
        boxShadow: isDark
          ? 'inset 0 1px 0 rgba(255,255,255,0.14), inset 0 -1px 0 rgba(0,0,0,0.45), inset 0 0 0 1px rgba(0,0,0,0.5), 0 3px 8px rgba(0,0,0,0.55)'
          : 'inset 0 1px 0 rgba(255,255,255,0.85), inset 0 -1px 0 rgba(0,0,0,0.1), inset 0 0 0 1px rgba(0,0,0,0.08), 0 3px 7px rgba(40,30,60,0.13)' };

  const fs = (typeof size === 'number' ? size : 200) > 60 ? '22px' : '12px';

  if (coverPath) {
    // `eager` marks the player's hero cover (and its transition snapshot) —
    // the one place that needs the full-resolution original. Everything else
    // (list rows, grids, chips) renders small → server thumbnail.
    const fullSrc = coverPath.startsWith('http') ? coverPath : `${API}${coverPath}`;
    const imgSrc = eager ? fullSrc : thumbCoverUrl(fullSrc);
    return (
      <div style={boxStyle}>
        <img
          src={imgSrc} alt=""
          // Native lazy-load: the browser only fetches covers in/near the
          // viewport. Off-screen/hidden covers must never compete with the
          // audio stream for the connection pool.
          loading={eager ? 'eager' : 'lazy'}
          decoding={eager ? 'sync' : 'async'}
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          onError={e => {
            e.currentTarget.style.display = 'none';
            const parent = e.currentTarget.parentElement;
            const span = document.createElement('span');
            span.style.cssText = `display:flex;align-items:center;justify-content:center;width:100%;height:100%;
              background:linear-gradient(135deg, oklch(38% 0.13 ${hue}), oklch(52% 0.18 ${(hue+45)%360}));
              font-size:${fs};font-weight:700;color:rgba(255,255,255,0.65);
              font-family:'Geist', 'Inter', system-ui, sans-serif;letter-spacing:0.5px;`;
            span.textContent = (title||'?').slice(0,2).toUpperCase();
            parent.appendChild(span);
          }}
        />
      </div>
    );
  }
  return (
    <div style={{
      ...boxStyle,
      background: `linear-gradient(135deg, oklch(38% 0.13 ${hue}), oklch(52% 0.18 ${(hue+45)%360}))`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: fs, fontWeight: '700',
      color: 'rgba(255,255,255,0.65)', fontFamily: "'Geist', 'Inter', system-ui, sans-serif",
      letterSpacing: '0.5px',
    }}>
      {(title||'?').slice(0,2).toUpperCase()}
    </div>
  );
}

// ─── LAZY COVER ───────────────────────────────────────────────────────────────
// CSS background-image covers download EAGERLY the moment the element renders
// (even in a hidden section) — loading="lazy" only exists on a real <img>.
// List artwork goes through this so off-screen covers never compete with the
// audio stream for the 6-per-origin connection pool (same reasoning as the
// loading="lazy" in AlbumCover above). No artwork → a plain gradient div.
/** Лениво загружаемая обложка по URL; без url — фирменный градиент-заглушка. Ошибка загрузки откатывается на градиент, а не на «битую» иконку. */
export function LazyCover({ url, className, style, fallback = 'linear-gradient(135deg, rgba(124,91,255,.35) 0%, rgba(255,120,200,.25) 100%)' }) {
  // Key order matters in the fallback: `background` (shorthand) spread AFTER
  // style wins over any backgroundColor placeholder carried in `style`.
  if (!url) return <div className={className} style={{ ...style, background: fallback }} />;
  return (
    <img src={thumbCoverUrl(url)} alt="" loading="lazy" decoding="async" className={className}
         // The gradient doubles as a loading placeholder behind the not-yet-
         // decoded image; a 404 swaps in a transparent pixel (data: can't
         // re-error) so the gradient shows instead of a broken-image glyph.
         style={{ objectFit: 'cover', background: fallback, ...style }}
         onError={e => { e.currentTarget.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'; }} />
  );
}

// ─── MOSAIC COVER ─────────────────────────────────────────────────────────────
/** Мозаика из 1–4 обложек (сетка 2×2) для плейлистов; пустой список — градиент с нотой. size число (px) или CSS-строка. */
export function MosaicCover({ trackIds = [], coverPaths = [], size = 220, radius = 12 }) {
  const n = Math.min(trackIds.length, 4);
  // A numeric size is a fixed px square. A flexible size (e.g. '100%') would
  // collapse to 0 height if we naively set height:'100%' and the parent has no
  // defined height (the playlist card's relative wrapper is auto-height) — so
  // for non-numeric sizes we enforce squareness with aspect-ratio instead.
  const dim = typeof size === 'number'
    ? { width: size, height: size }
    : { width: size, aspectRatio: '1 / 1' };
  if (n === 0) {
    return (
      <div style={{
        ...dim, borderRadius: radius, overflow: 'hidden',
        background: 'linear-gradient(135deg, rgba(124,91,255,.22) 0%, rgba(255,120,200,.14) 100%)',
        display: 'grid', placeItems: 'center', color: 'rgba(255,255,255,.45)',
        fontSize: typeof size === 'number' ? Math.round(size * 0.34) : 36,
      }}>♫</div>
    );
  }

  const grid = {
    ...dim, borderRadius: radius, overflow: 'hidden',
    display: 'grid', gap: 1, background: 'rgba(0,0,0,.4)',
  };
  if (n === 1) Object.assign(grid, { gridTemplateColumns: '1fr', gridTemplateRows: '1fr' });
  if (n === 2) Object.assign(grid, { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr' });
  if (n === 3) Object.assign(grid, { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr' });
  if (n === 4) Object.assign(grid, { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr' });

  return (
    <div style={grid}>
      {Array.from({ length: n }).map((_, i) => (
        <LazyCover key={i}
          url={coverPaths[i] ? `${API}${coverPaths[i]}` : null}
          style={{
            width: '100%', height: '100%', minWidth: 0, minHeight: 0, display: 'block',
            backgroundColor: 'rgba(255,255,255,.04)',
            ...(n === 3 && i === 0 ? { gridColumn: '1 / -1' } : {}),
          }}
        />
      ))}
    </div>
  );
}

// ─── ToggleSwitch (rocker button) ─────────────────────────────────────────────
/** Двухпозиционный «рокер»-переключатель: активная половина подсвечивается фирменным градиентом. */
export function ToggleSwitch({ checked, onChange, isDark }) {
  const leftActive = !checked;
  const rightActive = checked;
  const active = 'linear-gradient(180deg, oklch(67% 0.18 270), oklch(54% 0.22 280))';
  const activeShadow = 'inset 0 1px 3px rgba(0,0,0,0.35), 0 0 8px oklch(60% 0.18 270 / 0.4)';
  const inactiveD = {
    background: 'linear-gradient(180deg, #2c2c34, #1c1c23)',
    boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.12), 0 1px 2px rgba(0,0,0,0.4)',
  };
  const inactiveL = {
    background: 'linear-gradient(180deg, #fff, #e1e0ea)',
    boxShadow: 'inset 0 1px 0 rgba(255,255,255,1), 0 1px 2px rgba(40,30,60,0.1)',
  };

  return (
    <button
      onClick={() => onChange(!checked)}
      style={{
        width: '44px', height: '22px', borderRadius: '8px',
        display: 'flex', flexShrink: 0, padding: '2px',
        background: isDark ? '#15151a' : '#d8d7df',
        boxShadow: isDark
          ? 'inset 0 2px 4px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.04)'
          : 'inset 0 2px 4px rgba(40,30,60,0.15), 0 1px 0 rgba(255,255,255,0.8)',
        border: 'none', cursor: 'pointer', gap: '2px',
      }}
    >
      <span style={{
        flex: 1, borderRadius: '6px', transition: 'all 0.2s',
        ...(leftActive
          ? { background: active, boxShadow: activeShadow }
          : (isDark ? inactiveD : inactiveL)),
      }} />
      <span style={{
        flex: 1, borderRadius: '6px', transition: 'all 0.2s',
        ...(rightActive
          ? { background: active, boxShadow: activeShadow }
          : (isDark ? inactiveD : inactiveL)),
      }} />
    </button>
  );
}

// ─── Knob (skeuomorphic toggle) ───────────────────────────────────────────────
/** Круглая скеуоморфная ручка-крутилка с меткой под углом angle; glow подсвечивает метку янтарным. Используется, например, как переключатель темы. */
export function Knob({ angle = -45, size = 44, isDark, label, onClick, glow }) {
  const c = useColors(isDark);
  return (
    <button
      onClick={onClick}
      title={label}
      style={{
        width: size, height: size, borderRadius: '50%', flexShrink: 0,
        position: 'relative', cursor: 'pointer', padding: 0,
        background: isDark
          ? 'radial-gradient(circle at 30% 25%, #38383f 0%, #1a1a20 65%, #0c0c10 100%)'
          : 'radial-gradient(circle at 30% 25%, #ffffff 0%, #d8d7df 65%, #b8b7c1 100%)',
        boxShadow: isDark
          ? 'inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -1px 0 rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.55), 0 0 0 1px rgba(0,0,0,0.6)'
          : 'inset 0 1px 0 rgba(255,255,255,1), inset 0 -1px 0 rgba(0,0,0,0.13), 0 2px 5px rgba(40,30,60,0.18), 0 0 0 1px rgba(0,0,0,0.1)',
      }}
    >
      <div className="knob" style={{
        position:'absolute', inset:'0', borderRadius:'50%',
        transform:`rotate(${angle}deg)`,
      }}>
        <div style={{
          position:'absolute', top:'12%', left:'50%', transform:'translateX(-50%)',
          width:'2.5px', height:'30%', borderRadius:'2px',
          background: glow ? c.amber : (isDark ? '#9a99a4' : '#6e6c7a'),
          boxShadow: glow ? `0 0 8px ${c.amberGlow}` : 'none',
        }} />
      </div>
    </button>
  );
}

// ─── ResultCard ───────────────────────────────────────────────────────────────
/** Строка результата поиска: обложка, название/артист/год, бейдж «score · режим» (lyrics/audio/hybrid) и опциональная кнопка Play. c — объект useColors. */
export function ResultCard({ hit, isDark, c, compact=false, onClick, onPlay }) {
  const track = hit.track || {};
  const modeColor = { lyrics: c.accent, audio: 'oklch(60% 0.18 310)', hybrid: c.amber }[hit.matched_on] || c.accent;
  return (
    <div
      onClick={onClick}
      onMouseEnter={e => { e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)'; }}
      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
      style={{ display:'flex', alignItems:'center', gap:'12px', padding: compact?'9px 12px':'12px 14px',
        borderRadius:'12px', cursor: onClick?'pointer':'default', transition:'background 0.15s' }}
    >
      <AlbumCover title={track.title} artist={track.artist} size={compact?40:50} isDark={isDark} coverPath={track.cover_art_path} />
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontSize:'14px', fontWeight:'600', color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', letterSpacing:'-0.01em' }}>{track.title||'—'}</div>
        <div style={{ fontSize:'14px', color:c.textMuted, marginTop:'2px' }}>{track.artist||'—'}{track.year?` · ${track.year}`:''}</div>
      </div>
      {onPlay && (
        <button onClick={(e) => { e.stopPropagation(); onPlay(hit); }}
          title="Play"
          style={{
            width:'34px', height:'34px', borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center',
            flexShrink:0, background:'linear-gradient(180deg, oklch(60% 0.21 270), oklch(48% 0.22 285))',
            boxShadow:'inset 0 1px 0 rgba(255,255,255,0.2), inset 0 -1px 0 rgba(0,0,0,0.25), 0 2px 8px oklch(58% 0.21 270 / 0.4)',
            fontSize:'13px', color:'white',
          }}>
          ▶
        </button>
      )}
      <div style={{ display:'flex', flexDirection:'column', alignItems:'flex-end', gap:'4px', flexShrink:0 }}>
        <div className="mono" style={{ padding:'3px 9px', borderRadius:'20px', fontSize:'14px', fontWeight:'500',
          background: modeColor.replace(')', ' / 0.14)'),
          color: modeColor, border: `1px solid ${modeColor.replace(')', ' / 0.3)')}` }}>
          {Math.round((hit.score||0)*100)}% · {hit.matched_on}
        </div>
        {track.genre && <span className="mono" style={{ fontSize:'14px', color:c.textSubtle }}>{track.genre}</span>}
      </div>
    </div>
  );
}

// ─── Brand mark ───────────────────────────────────────────────────────────────
const BRAND_EQ_BARS = [
  { h: 0.44, c: 'oklch(64% 0.19 268)' },
  { h: 0.70, c: 'oklch(69% 0.17 278)' },
  { h: 0.94, c: 'oklch(74% 0.14 292)' },
  { h: 0.60, c: 'oklch(72% 0.09 350)' },
  { h: 0.36, c: 'oklch(76% 0.13 80)' },
];
/** Логотип MusiX: тёмная плашка с пятью EQ-столбиками фирменных цветов. */
export function BrandMark({ size=34, isDark }) {
  const barW = 3.1, gap = 1.3;
  const totalW = BRAND_EQ_BARS.length * barW + (BRAND_EQ_BARS.length - 1) * gap;
  const startX = (24 - totalW) / 2;
  return (
    <div style={{
      width:size, height:size, borderRadius:`${size*0.28}px`, flexShrink:0, position:'relative',
      background:'linear-gradient(150deg, #211c30, #0c0a13)',
      display:'flex', alignItems:'center', justifyContent:'center',
      boxShadow: isDark
        ? 'inset 0 1px 0 rgba(255,255,255,0.12), inset 0 -1px 0 rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,0,0,0.4)'
        : 'inset 0 1px 0 rgba(255,255,255,0.16), inset 0 -1px 0 rgba(0,0,0,0.3), 0 3px 10px rgba(0,0,0,0.35), 0 0 0 1px rgba(0,0,0,0.08)',
    }}>
      <svg width={size*0.6} height={size*0.6} viewBox="0 0 24 24" fill="none">
        {BRAND_EQ_BARS.map((b, i) => {
          const h = b.h * 19;
          const x = startX + i * (barW + gap);
          const y = 12 - h / 2;
          return <rect key={i} x={x} y={y} width={barW} height={h} rx={barW/2} fill={b.c} />;
        })}
      </svg>
    </div>
  );
}

// Shimmer skeleton block. Sizes accept px numbers or CSS strings (e.g. '80%').
/** Скелетон-блок с шиммером (классы .load-skel из index.css). w/h/r — px или CSS-строки. */
export function Skel({ w, h, r, isDark, style }) {
  return (
    <div className={isDark ? 'load-skel' : 'load-skel load-skel--l'}
         style={{ width:w, height:h, borderRadius:(r==null?8:r), ...(style||{}) }} />
  );
}

// ─── HintBadge ────────────────────────────────────────────────────────────────
/** Значок «i» с подсказкой: на десктопе hover-тултип, на мобильном — bottom sheet через портал. label может быть JSX. */
export function HintBadge({ size = 20, label, ariaLabel, placement = 'up' }) {
  const px = Math.max(14, size);
  const isMobile = useIsMobile();
  const [open, setOpen] = useState(false);
  const a11y = ariaLabel || (typeof label === 'string' ? label : 'Info');

  // Mobile: the hover tooltip both can't hover and used to spill past the
  // screen edge. Tap opens a bottom sheet instead (portal — the badge often
  // sits inside overflow-clipped rows).
  if (isMobile) {
    return (
      <span className="hint-badge" role="note" aria-label={a11y}
        tabIndex={0} onClick={() => setOpen(true)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(true); } }}>
        <span className="hint-badge__dot" aria-hidden="true"
          style={{ width: px, height: px, fontSize: Math.max(10, Math.round(px * 0.58)) }}>i</span>
        {open && createPortal(
          <div className="hint-sheet-scrim" onClick={(e) => { e.stopPropagation(); setOpen(false); }}>
            <div className="hint-sheet" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
              <span className="hint-sheet__grip" aria-hidden="true" />
              <div className="hint-sheet__body">{label}</div>
            </div>
          </div>,
          document.body
        )}
      </span>
    );
  }

  return (
    <span className={`hint-badge${placement === 'down' ? ' hint-badge--down' : ''}`}
      tabIndex={0} role="note"
      aria-label={a11y}>
      <span className="hint-badge__dot" aria-hidden="true"
        style={{ width: px, height: px, fontSize: Math.max(10, Math.round(px * 0.58)) }}>i</span>
      <span className="hint-badge__pop" role="tooltip">{label}</span>
    </span>
  );
}

// ─── SectionHeader ────────────────────────────────────────────────────────────
/** Шапка секции: тонкая градиентная полоса с нижней границей; right — слот для табов/фильтров, без right и children коллапсирует в null. */
export function SectionHeader({ isDark, lang, kicker, title, accent, right, children }) {
  const c = useColors(isDark);
  // The kicker + big serif title were removed app-wide to free vertical
  // space (sidebar nav already shows which section is active). The header
  // remains only as a host for `right` (tab toggles, filters) and `children`.
  // When nothing fills those slots, the component collapses entirely.
  if (!right && !children) return null;
  return (
    <div style={{ padding:'14px 36px 10px', borderBottom:`1px solid ${c.border}`,
      background: isDark ? 'linear-gradient(180deg, rgba(255,255,255,0.018) 0%, transparent 100%)' : 'linear-gradient(180deg, rgba(255,255,255,0.6) 0%, transparent 100%)' }}>
      {right && (
        <div style={{ display:'flex', justifyContent:'flex-end' }}>
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

// ─── Segmented control ────────────────────────────────────────────────────────
/** Сегментированный контрол во «врезанном» жёлобе: активный сегмент — рельефная кнопка. options: [{value, label}]. */
export function Segmented({ value, onChange, options, isDark, size='md', style } = {}) {
  const c = useColors(isDark);
  const padY = size === 'sm' ? '5px' : '7px';
  return (
    <div className={ske('inset', isDark)} style={{ display:'inline-flex', padding:'3px', borderRadius:'10px', gap:'2px', ...style }}>
      {options.map(o => {
        const active = value === o.value;
        return (
          <button key={o.value} onClick={() => onChange(o.value)}
            className={active ? ske('btn', isDark) : ''}
            style={{
              padding:`${padY} 14px`, borderRadius:'8px',
              fontSize: size === 'sm' ? '11px' : '12px', fontWeight:'600',
              fontFamily:"'Geist', 'Inter', system-ui, sans-serif", letterSpacing:'0.06em',
              color: active ? c.text : c.textSubtle,
              transition:'all 0.18s', whiteSpace:'nowrap',
            }}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Галочка-чек (SVG) для шагов мастеров и списков «сделано». */
export const StepCheck = ({ size = 10 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
);

// Crossfades its text on change: the outgoing label floats up and out while
// the incoming one rises in (absolute overlap — never a hard swap).
/** Кроссфейд текста при смене: старая подпись уплывает вверх, новая поднимается (классы .xfade из index.css). */
export function CrossfadeText({ text }) {
  const [pair, setPair] = useState({ curr: text, prev: null, k: 0 });
  useEffect(() => {
    setPair(p => (text === p.curr ? p : { curr: text, prev: p.curr, k: p.k + 1 }));
  }, [text]);
  return (
    <span className="xfade-wrap">
      {pair.prev != null && (
        <span key={`p${pair.k}`} className="xfade out"
          onAnimationEnd={() => setPair(p => ({ ...p, prev: null }))}>{pair.prev}</span>
      )}
      <span key={`c${pair.k}`} className="xfade in">{pair.curr}</span>
    </span>
  );
}

/** Курсивная заглушка «Нет данных / No data». */
export function Empty({ lang }) {
  return <div style={{ fontSize:11, fontStyle:'italic', opacity:0.5 }}>{lang==='ru'?'Нет данных':'No data'}</div>;
}

// Labelled seam between stats sections (reuses the .rec-div divider language).
/** Подписанный шов-разделитель между секциями: светящаяся точка + градиентные линии, цвет задаётся hue (oklch). */
export function StatsDivider({ label, hue }) {
  const col = `oklch(70% 0.15 ${hue})`;
  const ln = { background:`linear-gradient(90deg,transparent,oklch(70% 0.15 ${hue} / .45),transparent)` };
  return (
    <div className="rec-div" style={{ margin:'6px 4px 0' }}>
      <div className="rec-div__ln" style={ln} />
      <div className="rec-div__lbl" style={{ color: col, fontSize:'clamp(12px, 1.1vw, 14px)' }}>
        <span className="rec-div__nd" style={{ background: col, boxShadow:`0 0 8px ${col}` }} />
        {label}
      </div>
      <div className="rec-div__ln" style={ln} />
    </div>
  );
}

// ─── Sparkline ────────────────────────────────────────────────────────────────
/** Мини-график активности за 14 дней: линия + градиентная заливка. days: [{date:'YYYY-MM-DD', count}], hue — oklch-тон. */
export function Sparkline({ days, width=92, height=24, hue=145 }) {
  const pts = useMemo(() => {
    const map = new Map((days||[]).map(d => [d.date, d.count]));
    const today = new Date(); today.setHours(0,0,0,0);
    const vals = [];
    for (let i=13;i>=0;i--){
      const dt = new Date(today); dt.setDate(today.getDate()-i);
      const iso = `${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,'0')}-${String(dt.getDate()).padStart(2,'0')}`;
      vals.push(map.get(iso) || 0);
    }
    return vals;
  }, [days]);
  const max = Math.max(1, ...pts);
  const stepX = width/(pts.length-1);
  const line = pts.map((v,i)=>`${i?'L':'M'}${(i*stepX).toFixed(1)},${(height-(v/max)*(height-3)-1).toFixed(1)}`).join(' ');
  const area = `${line} L${width},${height} L0,${height} Z`;
  const gid = `spk${hue}`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display:'block', flex:'none' }}>
      <defs><linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor={`oklch(72% 0.16 ${hue})`} stopOpacity="0.32" />
        <stop offset="100%" stopColor={`oklch(72% 0.16 ${hue})`} stopOpacity="0" />
      </linearGradient></defs>
      <path d={area} fill={`url(#${gid})`} />
      <path d={line} fill="none" stroke={`oklch(74% 0.16 ${hue})`} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// Glass readout chip used in the Rhythm section header (streak / busiest day).
// Optional `foot` adds a clarifying third line.
/** Стеклянный чип-показание: эмодзи-иконка, крупное значение, tracked-caps подпись цвета hue; foot — третья поясняющая строка. */
export function RhythmReadout({ icon, value, label, foot, isDark, hue, grow }) {
  const c = useColors(isDark);
  const glass = isDark ? {
    background:'linear-gradient(165deg, rgba(255,255,255,0.13), rgba(255,255,255,0.03)), rgba(24,24,32,0.36)',
    border:'1px solid rgba(255,255,255,0.13)',
    boxShadow:'inset 0 1px 0 rgba(255,255,255,0.22), 0 10px 26px rgba(0,0,0,0.32)',
  } : {
    background:'linear-gradient(165deg, rgba(255,255,255,0.92), rgba(255,255,255,0.5)), rgba(244,243,250,0.4)',
    border:'1px solid rgba(255,255,255,0.8)',
    boxShadow:'inset 0 1px 0 rgba(255,255,255,0.95), 0 10px 26px rgba(46,36,86,0.12)',
  };
  return (
    <div className={grow ? 'rhythm-readout rhythm-readout--grow' : 'rhythm-readout'}
      style={{ ...glass, flex: grow?'1 1 280px':'0 1 auto', display:'flex', alignItems:'center', gap:14,
      padding:'14px 18px', borderRadius:15, minWidth:0, backdropFilter:'blur(14px) saturate(1.5)', WebkitBackdropFilter:'blur(14px) saturate(1.5)' }}>
      <span style={{ fontSize:'clamp(22px, 2.2vw, 28px)', flex:'none', lineHeight:1 }}>{icon}</span>
      <div style={{ minWidth:0 }}>
        <div style={{ fontSize:'clamp(18px, 1.9vw, 23px)', fontWeight:700, color:c.text, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{value}</div>
        <div className="mono" style={{ fontSize:'clamp(10px, 1vw, 12px)', letterSpacing:'0.1em', textTransform:'uppercase', color:`oklch(70% 0.13 ${hue})`, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{label}</div>
        {foot && <div style={{ fontSize:'clamp(12px, 1.05vw, 13.5px)', color:c.textMuted, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', marginTop:3 }}>{foot}</div>}
      </div>
    </div>
  );
}

// ─── SkeuoArcGauge ────────────────────────────────────────────────────────────
/** Полукруглый шкальный индикатор (0..1) со светящейся дугой цвета hue и процентом под ней; анимируется при попадании в вьюпорт, уважает reduced-motion. */
export function SkeuoArcGauge({ value=0, hue=145, label, sub, isDark }) {
  const c = useColors(isDark);
  const ref = useRef(null);
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState(reduced);
  useEffect(() => {
    if (reduced) { setShown(true); return; }
    const el = ref.current; if (!el) return;
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setShown(true); io.disconnect(); } }, { threshold: 0.4 });
    io.observe(el); return () => io.disconnect();
  }, [reduced]);
  const pct = Math.max(0, Math.min(1, value));
  const r=54, cx=64, cy=64, len = Math.PI * r;
  const fill = shown ? pct : 0;
  const a = Math.PI - pct*Math.PI;
  const [hx, hy] = shown ? [cx + r*Math.cos(a), cy - r*Math.sin(a)] : [cx - r, cy];
  return (
    <div ref={ref} style={{ textAlign:'center', flex:'none' }}>
      <svg viewBox="0 0 128 78" width="168" height="103">
        <path d="M10,64 A54,54 0 0 1 118,64" fill="none"
          stroke={isDark?'rgba(0,0,0,.5)':'rgba(40,30,60,.16)'} strokeWidth="9" strokeLinecap="round" />
        <path d="M10,64 A54,54 0 0 1 118,64" fill="none"
          stroke={`oklch(70% 0.17 ${hue})`} strokeWidth="9" strokeLinecap="round"
          strokeDasharray={len} strokeDashoffset={len*(1-fill)}
          style={{ transition: reduced?'none':'stroke-dashoffset 0.9s cubic-bezier(.22,.9,.3,1)',
                   filter:`drop-shadow(0 0 6px oklch(70% 0.17 ${hue} / .5))` }} />
        <circle cx={hx} cy={hy} r="4.5" fill="#fff"
          style={{ filter:`drop-shadow(0 0 5px oklch(75% 0.18 ${hue}))`,
                   transition: reduced?'none':'cx 0.9s cubic-bezier(.22,.9,.3,1), cy 0.9s cubic-bezier(.22,.9,.3,1)' }} />
      </svg>
      <div style={{ fontSize:'clamp(28px, 3vw, 36px)', fontWeight:800, color:c.text, marginTop:-12 }}>{Math.round(pct*100)}%</div>
      {label && <div className="mono" style={{ fontSize:'clamp(10px, 1vw, 12px)', letterSpacing:'0.2em', textTransform:'uppercase', color:c.textSubtle }}>{label}</div>}
      {sub && <div style={{ fontSize:'clamp(12px, 1.05vw, 13.5px)', color:c.textMuted, marginTop:3 }}>{sub}</div>}
    </div>
  );
}

// Small embossed completion dial for the "loved" list rows.
/** Компактное кольцо завершённости (0..1) с числом процентов в центре, цвет — hue. */
export function CompletionRing({ pct=0, size=42, hue=145, isDark }) {
  const c = useColors(isDark);
  const r=(size-7)/2, cc=size/2, len=2*Math.PI*r, p=Math.max(0,Math.min(1,pct));
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ flex:'none' }}>
      <circle cx={cc} cy={cc} r={r} fill="none" stroke={isDark?'rgba(255,255,255,.1)':'rgba(0,0,0,.1)'} strokeWidth="3.5" />
      <circle cx={cc} cy={cc} r={r} fill="none" stroke={`oklch(70% 0.16 ${hue})`} strokeWidth="3.5" strokeLinecap="round"
        strokeDasharray={len} strokeDashoffset={len*(1-p)} transform={`rotate(-90 ${cc} ${cc})`}
        style={{ filter:`drop-shadow(0 0 3px oklch(70% 0.16 ${hue} / .5))` }} />
      <text x={cc} y={cc+4} textAnchor="middle" style={{ fontSize:11, fontWeight:700, fill:c.text }}>{Math.round(p*100)}</text>
    </svg>
  );
}

// ─── InfoTip ──────────────────────────────────────────────────────────────────
/** Маленький значок «i» с всплывающей карточкой-подсказкой по hover/фокусу (250px, тема-зависимая). */
export function InfoTip({ isDark, text, size = 15, style }) {
  const [open, setOpen] = useState(false);
  return (
    <span style={{ position:'relative', display:'inline-flex', ...style }}
      onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} tabIndex={0}>
      <span aria-hidden style={{
        width:size, height:size, borderRadius:'50%', flexShrink:0, cursor:'help',
        display:'inline-flex', alignItems:'center', justifyContent:'center',
        fontSize: size*0.62, fontStyle:'italic', fontWeight:700, lineHeight:1,
        color: isDark ? 'rgba(255,255,255,0.75)' : 'rgba(0,0,0,0.6)',
        background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.07)',
      }}>i</span>
      {open && (
        <span style={{
          position:'absolute', bottom:'calc(100% + 8px)', left:'50%', transform:'translateX(-50%)',
          width:'250px', maxWidth:'70vw', padding:'10px 12px', borderRadius:'10px', zIndex:60,
          background: isDark ? '#23232b' : '#ffffff',
          color: isDark ? '#d8dbe3' : '#333',
          border:`1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
          boxShadow:'0 10px 34px rgba(0,0,0,0.28)',
          fontSize:'12px', lineHeight:1.5, fontWeight:400, letterSpacing:'normal',
          textTransform:'none', textAlign:'left', pointerEvents:'none',
        }}>
          {text}
        </span>
      )}
    </span>
  );
}

// ─── OBStageBar ───────────────────────────────────────────────────────────────
/** Строка прогресса этапа (онбординг/индексация): подпись, статус ✓/✗/…/·, полоса с процентом или indeterminate-шиммером (.ob-indet). c — объект useColors. */
export function OBStageBar({ c, label, state, pct, indeterminate, eta, count, trailing }) {
  return (
    <div style={{ marginBottom:12 }}>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
        <span style={{ display:'flex', alignItems:'center', gap:8, minWidth:0 }}>
          <span className="mono" style={{ fontSize:12, letterSpacing:'0.12em', color: state==='running' ? c.text : c.textSubtle }}>{label}</span>
          {trailing}
        </span>
        <span style={{ display:'flex', alignItems:'center', gap:8, fontSize:11, color: state==='failed' ? c.red : c.textSubtle }}>
          {count && state==='running' && <span className="mono" style={{ fontVariantNumeric:'tabular-nums', opacity:.95 }}>{count}</span>}
          {eta && state==='running' && <span style={{ fontVariantNumeric:'tabular-nums', opacity:.85 }}>{eta}</span>}
          <span>{state==='done' ? '✓' : state==='failed' ? '✗' : state==='running' ? '…' : '·'}</span>
        </span>
      </div>
      <div style={{ height:5, borderRadius:3, background:c.border, overflow:'hidden', position:'relative' }}>
        {indeterminate && state==='running'
          ? <div className="ob-indet" />
          : <div style={{ height:'100%', width:`${pct||0}%`, transition:'width 0.4s', background: state==='done' ? 'oklch(63% 0.17 142)' : 'linear-gradient(90deg, oklch(65% 0.18 270), oklch(75% 0.17 280))' }} />}
      </div>
    </div>
  );
}

// ─── Onboarding glass cards ───────────────────────────────────────────────────
/** Стеклянная карточка онбординга (класс .ob-glass): блюр, светлая рамка, мягкая тень. */
export function GlassCard({ children, style, className }) {
  return <div className={'ob-glass' + (className ? ' ' + className : '')} style={style}>{children}</div>;
}

/** Карточка фичи: крупная эмодзи-иконка, заголовок, описание; premium-вариант с золотой рамкой (.ob-feat-premium). */
export function FeatureCard({ icon, title, body, premium }) {
  return (
    <div className={'ob-feat ' + (premium ? 'ob-feat-premium' : 'ob-feat-hover')} style={{ padding:'clamp(15px, 1.3vw, 26px)' }}>
      <div style={{ fontSize:'clamp(20px, 1.7vw, 32px)', lineHeight:1 }}>{icon}</div>
      <div style={{ fontWeight:600, fontSize:'clamp(14px, 1.05vw, 20px)', margin:'10px 0 5px' }}>{title}</div>
      <div style={{ fontSize:'clamp(12.5px, 0.92vw, 16px)', lineHeight:1.45, opacity:.72 }}>{body}</div>
    </div>
  );
}

/** Выбираемая карточка режима (мастер настройки): sel подсвечивает фиолетовой рамкой; note — сноска в сиреневой плашке. Требует CSS-переменные --ob-card-bg/--ob-card-edge на родителе. */
export function ModeCard({ sel, onClick, title, body, note }) {
  return (
    <div onClick={onClick} className="ob-hoverlift" style={{ cursor:'pointer', background:'var(--ob-card-bg)', borderRadius:12, padding:13,
        boxShadow: sel?'inset 0 0 0 2px oklch(62% .2 275),0 6px 18px rgba(124,92,255,.3)':'inset 0 0 0 1px var(--ob-card-edge),0 3px 10px rgba(0,0,0,.18)' }}>
      <div style={{ fontWeight:600, fontSize:13 }}>{title}</div>
      <div style={{ fontSize:11.5, opacity:.7, lineHeight:1.4, marginTop:3 }}>{body}</div>
      {note && <div style={{ fontSize:10.5, color:'#cdbfff', background:'rgba(139,116,255,.12)', border:'1px solid rgba(139,116,255,.3)', borderRadius:8, padding:'6px 9px', marginTop:8 }}>{note}</div>}
    </div>
  );
}
