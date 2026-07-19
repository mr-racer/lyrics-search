# MusiX UI — как строить с этой системой

**Эстетика:** «студийная консоль» — скеуоморфные панели, врезанные жёлобы, мягкие свечения, oklch-акценты. Тёмная тема — основная (фон страницы `#0d0d10`, светлая — `#f2f1f6`).

## Подключение и темизация
Провайдер не нужен. Почти каждый компонент принимает `isDark: boolean`; палитру бери из `useColors(isDark)` — это JS-объект токенов (НЕ CSS-переменные): `bg, bgDeep, surface..surface3, border, borderStrong, text, textMuted, textSubtle, accent, accentBg, accentLight, amber, amberGlow, green, red, …`. Для светлой темы дополнительно ставь `document.body.dataset.theme = 'light'` — часть CSS (слайдер SkeRange, фокус-кольца) переключается селектором `body[data-theme="light"]`.

`GlassCard` / `FeatureCard` / `ModeCard` читают CSS-переменные с предка — задай на обёртке:
```jsx
<div style={{ '--ob-glass-bg':'rgba(255,255,255,.055)', '--ob-glass-sheen':'rgba(255,255,255,.12)',
  '--ob-glass-edge':'rgba(255,255,255,.18)', '--ob-card-bg':'linear-gradient(180deg,#1d1d23,#131318)',
  '--ob-card-edge':'rgba(255,255,255,.05)' }}>…</div>
```

## Идиома стилей
Инлайн-стили + небольшой набор реальных классов из `styles.css` (никакого Tailwind/CSS-modules; новые классы не выдумывай):
- Поверхности: `ske('panel'|'btn'|'inset'|'display', isDark)` → `ske-panel-d` и т.п.; акцентные кнопки `ske-accent` (фиолетовая) и `ske-amber` (янтарная, тёмный текст); металл `brushed(isDark)`; зерно `.grain`; стекло `.ob-glass`.
- Типографика: UI-шрифт Geist (грузится из styles.css, remote Google Fonts); `.mono` — tracked-caps подписи; `.serif` / `.vibe-serif` (Playfair) — display-заголовки; `.num-tab` — табличные цифры.
- Анимации уже в CSS: `spin`, `fadeIn`, `fadeInUp`, `shimmer`, `scaleIn`, `slideUp`; каскадное появление — класс `.stagger` на родителе.
- Скелетоны — компонент `Skel` (класс `load-skel`); модалки — `.modal-overlay` + `.modal-content`.
- Цвета пиши в oklch: акцент `oklch(60% 0.18 270)`, янтарь `oklch(72% 0.13 75)`, зелёный `oklch(63% 0.17 142)`, красный `oklch(58% 0.21 25)`.

## Где правда
Читай `styles.css` бандла (весь UI-язык: ske-поверхности, rec2-консоль, aio-орб, анимации) и `components/<группа>/<Имя>/<Имя>.d.ts` + `.prompt.md` — там точные пропсы и живые примеры каждого компонента.

## Идиоматичный пример
```jsx
const c = MusixUI.useColors(true);
<div className={MusixUI.ske('panel', true)} style={{ padding: 20, borderRadius: 16, color: c.text,
    fontFamily: "'Geist','Inter',system-ui,sans-serif" }}>
  <div className="mono" style={{ fontSize: 11, letterSpacing: '.18em', color: c.textSubtle }}>БИБЛИОТЕКА</div>
  <MusixUI.ResultCard isDark c={c}
    hit={{ score: .87, matched_on: 'lyrics', track: { title: 'Группа крови', artist: 'Кино', year: 1988 } }}
    onPlay={() => {}} />
  <MusixUI.SkeRange value={64} onChange={() => {}} accent={c.accent} />
</div>
```
Компоненты библиотеки — для контролов и виджетов; собственную обвязку (сетки, отступы) делай инлайн-стилями с токенами из `useColors`.
