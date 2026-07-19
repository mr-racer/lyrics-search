// Типы musix-ui. Пакет написан на JSX — этот файл ведётся вручную и является
// публичным контрактом API (его же читает design-sync для карточек компонентов).
import * as React from 'react';

/** Объект цветовых токенов MusiX (возвращается useColors). */
export interface Colors {
  bg: string; bgDeep: string; surface: string; surface2: string; surface3: string;
  border: string; borderStrong: string;
  text: string; textMuted: string; textSubtle: string;
  accent: string; accentBg: string; accentLight: string;
  amber: string; amberGlow: string;
  inputBg: string; sidebarBg: string; chatPanelBg: string;
  userBubble: string; aiBubble: string;
  green: string; greenBg: string; red: string; redBg: string;
}
/** Палитра MusiX для тёмной/светлой темы; компоненты берут цвета из неё. */
export function useColors(isDark?: boolean): Colors;
/** Имя скеуоморфного CSS-класса поверхности: ske('panel', true) → 'ske-panel-d'. */
export function ske(kind: 'panel' | 'btn' | 'inset' | 'display', isDark?: boolean): string;
/** Класс «шлифованный металл»: brushed(true) → 'brushed-d'. */
export function brushed(isDark?: boolean): string;
/** true, когда ОС просит уменьшить анимации (prefers-reduced-motion). */
export function usePrefersReducedMotion(): boolean;
/** true на мобильной ширине (≤768px). */
export function useIsMobile(): boolean;
/** URL серверной миниатюры обложки (?w=320). */
export function thumbCoverUrl(url: string, w?: number): string;

export interface SpinnerProps {
  /** Диаметр в px. */
  size?: number;
  /** Цвет дуги; по умолчанию фирменный фиолетовый oklch(60% 0.18 270). */
  color?: string;
}
/** Кольцевой индикатор загрузки. */
export function Spinner(props: SpinnerProps): React.JSX.Element;

export interface SkeRangeProps {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
  /** Цвет заливки/свечения (CSS color, обычно oklch). */
  accent?: string;
  /** Плавное скольжение ручки между фиксированными стопами. */
  animated?: boolean;
  /** Заливка растёт от нуля в центре (для −N..+N). */
  bipolar?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
  style?: React.CSSProperties;
}
/** Скеуоморфный слайдер: вырезанный жёлоб + рельефная ручка, нативный input сверху (drag/клавиатура/a11y). */
export function SkeRange(props: SkeRangeProps): React.JSX.Element;

export interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  isDark?: boolean;
}
/** Двухпозиционный «рокер»: активная половина светится фирменным градиентом. */
export function ToggleSwitch(props: ToggleSwitchProps): React.JSX.Element;

export interface KnobProps {
  /** Угол метки в градусах (−90..90). */
  angle?: number;
  size?: number;
  isDark?: boolean;
  /** title/подсказка кнопки. */
  label?: string;
  onClick?: () => void;
  /** Янтарная подсветка метки. */
  glow?: boolean;
}
/** Круглая скеуоморфная ручка-крутилка (например, переключатель темы). */
export function Knob(props: KnobProps): React.JSX.Element;

export interface SegmentedOption { value: string; label: React.ReactNode; }
export interface SegmentedProps {
  value: string;
  onChange: (value: string) => void;
  options: SegmentedOption[];
  isDark?: boolean;
  size?: 'sm' | 'md';
  style?: React.CSSProperties;
}
/** Сегментированный контрол во «врезанном» жёлобе; активный сегмент — рельефная кнопка. */
export function Segmented(props: SegmentedProps): React.JSX.Element;

export interface StepCheckProps { size?: number; }
/** Галочка-чек (SVG, stroke=currentColor) для шагов мастеров. */
export function StepCheck(props: StepCheckProps): React.JSX.Element;

export interface HintBadgeProps {
  size?: number;
  /** Содержимое подсказки (можно JSX). */
  label: React.ReactNode;
  ariaLabel?: string;
  /** Куда раскрывать десктопный тултип. */
  placement?: 'up' | 'down';
}
/** Значок «i»: hover-тултип на десктопе, bottom sheet на мобильном. */
export function HintBadge(props: HintBadgeProps): React.JSX.Element;

export interface InfoTipProps {
  isDark?: boolean;
  text: React.ReactNode;
  size?: number;
  style?: React.CSSProperties;
}
/** Маленький «i» с всплывающей карточкой-подсказкой по hover/фокусу. */
export function InfoTip(props: InfoTipProps): React.JSX.Element;

export interface SkelProps {
  w?: number | string;
  h?: number | string;
  r?: number | string;
  isDark?: boolean;
  style?: React.CSSProperties;
}
/** Скелетон-блок с шиммером (.load-skel). */
export function Skel(props: SkelProps): React.JSX.Element;

export interface CrossfadeTextProps { text: string; }
/** Кроссфейд подписи при смене текста (классы .xfade). */
export function CrossfadeText(props: CrossfadeTextProps): React.JSX.Element;

export interface AlbumCoverProps {
  title?: string;
  artist?: string;
  /** px или CSS-строка; влияет на радиус и размер инициалов. */
  size?: number | string;
  isDark?: boolean;
  /** Путь к арту (относительный к API или полный URL); без него — градиент с инициалами. */
  coverPath?: string;
  radius?: number;
  /** Растянуть на 100% контейнера. */
  fluid?: boolean;
  /** Полноразмерный арт с синхронной отрисовкой (hero-обложка плеера). */
  eager?: boolean;
}
/** Обложка альбома: изображение или детерминированный градиент с инициалами. */
export function AlbumCover(props: AlbumCoverProps): React.JSX.Element;

export interface LazyCoverProps {
  url?: string | null;
  className?: string;
  style?: React.CSSProperties;
  /** CSS background, показывается без url и на ошибке загрузки. */
  fallback?: string;
}
/** Лениво загружаемая обложка; без url — фирменный градиент. */
export function LazyCover(props: LazyCoverProps): React.JSX.Element;

export interface MosaicCoverProps {
  trackIds?: Array<string | number>;
  coverPaths?: Array<string | null>;
  size?: number | string;
  radius?: number;
}
/** Мозаика 1–4 обложек для плейлиста; пусто — градиент с нотой. */
export function MosaicCover(props: MosaicCoverProps): React.JSX.Element;

export interface BrandMarkProps { size?: number; isDark?: boolean; }
/** Логотип MusiX: тёмная плашка с пятью EQ-столбиками. */
export function BrandMark(props: BrandMarkProps): React.JSX.Element;

export interface Track {
  title?: string;
  artist?: string;
  year?: number;
  genre?: string;
  cover_art_path?: string;
}
export interface SearchHit {
  track?: Track;
  /** 0..1 — релевантность. */
  score?: number;
  matched_on?: 'lyrics' | 'audio' | 'hybrid';
}
export interface ResultCardProps {
  hit: SearchHit;
  isDark?: boolean;
  /** Палитра из useColors(isDark). */
  c: Colors;
  compact?: boolean;
  onClick?: () => void;
  /** Появляется круглая кнопка ▶. */
  onPlay?: (hit: SearchHit) => void;
}
/** Строка результата поиска: обложка, метаданные, бейдж «score · режим», Play. */
export function ResultCard(props: ResultCardProps): React.JSX.Element;

export interface SectionHeaderProps {
  isDark?: boolean;
  lang?: string;
  kicker?: React.ReactNode;
  title?: React.ReactNode;
  accent?: string;
  /** Слот справа (табы/фильтры); без right и children рендерится null. */
  right?: React.ReactNode;
  children?: React.ReactNode;
}
/** Шапка секции — тонкая градиентная полоса с нижней границей. */
export function SectionHeader(props: SectionHeaderProps): React.JSX.Element | null;

export interface GlassCardProps {
  children?: React.ReactNode;
  style?: React.CSSProperties;
  className?: string;
}
/** Стеклянная карточка онбординга (.ob-glass). */
export function GlassCard(props: GlassCardProps): React.JSX.Element;

export interface FeatureCardProps {
  icon: React.ReactNode;
  title: React.ReactNode;
  body?: React.ReactNode;
  /** Золотая premium-рамка (.ob-feat-premium). */
  premium?: boolean;
}
/** Карточка фичи: эмодзи-иконка, заголовок, описание. */
export function FeatureCard(props: FeatureCardProps): React.JSX.Element;

export interface ModeCardProps {
  /** Выбрана: фиолетовая рамка-свечение. */
  sel?: boolean;
  onClick?: () => void;
  title: React.ReactNode;
  body?: React.ReactNode;
  /** Сноска в сиреневой плашке. */
  note?: React.ReactNode;
}
/** Выбираемая карточка режима; требует CSS-переменные --ob-card-bg/--ob-card-edge на родителе. */
export function ModeCard(props: ModeCardProps): React.JSX.Element;

export interface EmptyProps { lang?: 'ru' | 'en'; }
/** Курсивная заглушка «Нет данных / No data». */
export function Empty(props: EmptyProps): React.JSX.Element;

export interface StatsDividerProps {
  label: React.ReactNode;
  /** oklch-тон (0..360) точки и линий. */
  hue: number;
}
/** Подписанный шов-разделитель со светящейся точкой. */
export function StatsDivider(props: StatsDividerProps): React.JSX.Element;

export interface SparklineDay { date: string; count: number; }
export interface SparklineProps {
  /** [{date:'YYYY-MM-DD', count}] за последние 14 дней (пропуски = 0). */
  days?: SparklineDay[];
  width?: number;
  height?: number;
  /** oklch-тон линии. */
  hue?: number;
}
/** Мини-график активности за 14 дней: линия + градиентная заливка. */
export function Sparkline(props: SparklineProps): React.JSX.Element;

export interface CompletionRingProps {
  /** 0..1. */
  pct?: number;
  size?: number;
  hue?: number;
  isDark?: boolean;
}
/** Кольцо завершённости с процентом в центре. */
export function CompletionRing(props: CompletionRingProps): React.JSX.Element;

export interface SkeuoArcGaugeProps {
  /** 0..1. */
  value?: number;
  hue?: number;
  label?: React.ReactNode;
  sub?: React.ReactNode;
  isDark?: boolean;
}
/** Полукруглый индикатор со светящейся дугой и крупным процентом. */
export function SkeuoArcGauge(props: SkeuoArcGaugeProps): React.JSX.Element;

export interface RhythmReadoutProps {
  icon: React.ReactNode;
  value: React.ReactNode;
  label: React.ReactNode;
  /** Третья поясняющая строка. */
  foot?: React.ReactNode;
  isDark?: boolean;
  hue?: number;
  /** flex-расти в ряду. */
  grow?: boolean;
}
/** Стеклянный чип-показание: иконка, крупное значение, tracked-caps подпись. */
export function RhythmReadout(props: RhythmReadoutProps): React.JSX.Element;

export interface OBStageBarProps {
  /** Палитра из useColors(isDark). */
  c: Colors;
  label: React.ReactNode;
  state?: 'pending' | 'running' | 'done' | 'failed';
  /** 0..100. */
  pct?: number;
  /** Шиммер вместо процента (.ob-indet). */
  indeterminate?: boolean;
  eta?: string;
  count?: string;
  trailing?: React.ReactNode;
}
/** Строка прогресса этапа: подпись, статус ✓/✗/…/·, полоса. */
export function OBStageBar(props: OBStageBarProps): React.JSX.Element;
