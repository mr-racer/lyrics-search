
// Stats Section — 2D scatter (PCA), genre/year/duration charts, top-5 similar/different pairs
const StatsSection = ({ theme, lang }) => {
  const isDark = theme === 'dark';

  const T = {
    ru: {
      title: 'Аналитика коллекции',
      subtitle: 'Твои треки в многомерном пространстве',
      scatter2d: '2D карта',
      scatter3d: '3D карта',
      genres: 'Жанры',
      years: 'Годы',
      duration: 'Длительность',
      colorBy: 'Подсветить по',
      byGenre: 'жанру',
      byArtist: 'исполнителю',
      byAlbum: 'альбому',
      byDecade: 'десятилетию',
      byDuration: 'длительности',
      similar: 'Топ-5 похожих по звуку',
      different: 'Топ-5 наиболее разных',
      tracks: 'треков',
      totalTracks: 'треков в коллекции',
      totalArtists: 'исполнителей',
      totalAlbums: 'альбомов',
      pairSimilarity: 'схожесть',
      pairDistance: 'расстояние',
      embedType: 'Эмбеддинги',
      textEmbed: 'Тексты',
      soundEmbed: 'Звук',
    },
    en: {
      title: 'Collection Analytics',
      subtitle: 'Your tracks in multidimensional space',
      scatter2d: '2D map',
      scatter3d: '3D map',
      genres: 'Genres',
      years: 'Years',
      duration: 'Duration',
      colorBy: 'Color by',
      byGenre: 'genre',
      byArtist: 'artist',
      byAlbum: 'album',
      byDecade: 'decade',
      byDuration: 'duration',
      similar: 'Top-5 most similar (sound)',
      different: 'Top-5 most different',
      tracks: 'tracks',
      totalTracks: 'tracks in collection',
      totalArtists: 'artists',
      totalAlbums: 'albums',
      pairSimilarity: 'similarity',
      pairDistance: 'distance',
      embedType: 'Embeddings',
      textEmbed: 'Lyrics',
      soundEmbed: 'Sound',
    }
  };
  const t = T[lang] || T.ru;

  const [scatterTab, setScatterTab] = React.useState('2d');
  const [colorBy, setColorBy] = React.useState('genre');
  const [embedType, setEmbedType] = React.useState('sound');
  const [statsTab, setStatsTab] = React.useState('genres');
  const [hoveredPoint, setHoveredPoint] = React.useState(null);
  const [tooltip, setTooltip] = React.useState(null);
  const svgRef = React.useRef(null);

  const genres = ['Rock', 'Electronic', 'Jazz', 'Classical', 'Hip-Hop', 'Folk'];
  const genreColors = [
    'oklch(60% 0.18 270)',
    'oklch(60% 0.18 200)',
    'oklch(62% 0.16 140)',
    'oklch(60% 0.14 60)',
    'oklch(60% 0.18 310)',
    'oklch(60% 0.14 20)',
  ];
  const decades = ["'60s", "'70s", "'80s", "'90s", "'00s", "'10s", "'20s"];
  const decadeColors = [
    'oklch(62% 0.16 30)',
    'oklch(60% 0.18 60)',
    'oklch(62% 0.16 100)',
    'oklch(60% 0.18 150)',
    'oklch(60% 0.18 200)',
    'oklch(60% 0.18 250)',
    'oklch(60% 0.18 310)',
  ];

  // Generate deterministic scatter points
  const scatterPoints = React.useMemo(() => {
    const pts = [];
    const seed = (n) => { let x = Math.sin(n) * 10000; return x - Math.floor(x); };
    const genreList = genres;
    for (let i = 0; i < 180; i++) {
      const gi = Math.floor(seed(i * 3.1) * genreList.length);
      const decade = Math.floor(seed(i * 2.7) * decades.length);
      // cluster by genre
      const cx = (gi / genreList.length) * 0.7 + 0.1;
      const cy = ((gi % 3) / 3) * 0.7 + 0.15;
      pts.push({
        id: i,
        x: cx + (seed(i * 7.3) - 0.5) * 0.22,
        y: cy + (seed(i * 5.1) - 0.5) * 0.22,
        genre: genreList[gi],
        genreIdx: gi,
        decade: decades[decade],
        decadeIdx: decade,
        title: `Track ${i+1}`,
        artist: ['Radiohead','Pink Floyd','Portishead','Massive Attack','Aphex Twin','Boards of Canada','Nick Drake','John Coltrane','Miles Davis','Bach'][Math.floor(seed(i*4.3)*10)],
        year: 1960 + Math.floor(seed(i*2.1) * 65),
        duration: `${Math.floor(seed(i*6.7)*5)+2}:${String(Math.floor(seed(i*8.3)*60)).padStart(2,'0')}`,
      });
    }
    return pts;
  }, []);

  const getPointColor = (pt) => {
    if (colorBy === 'genre') return genreColors[pt.genreIdx] || 'oklch(60% 0.1 270)';
    if (colorBy === 'decade') return decadeColors[pt.decadeIdx] || 'oklch(60% 0.1 270)';
    return 'oklch(60% 0.15 270)';
  };

  const genreCounts = React.useMemo(() => {
    const counts = {};
    genres.forEach(g => { counts[g] = 0; });
    scatterPoints.forEach(p => { if (counts[p.genre] !== undefined) counts[p.genre]++; });
    return genres.map((g, i) => ({ name: g, count: counts[g], color: genreColors[i] }));
  }, [scatterPoints]);

  const decadeCounts = React.useMemo(() => {
    const counts = {};
    decades.forEach(d => { counts[d] = 0; });
    scatterPoints.forEach(p => { counts[p.decade] = (counts[p.decade] || 0) + 1; });
    return decades.map((d, i) => ({ name: d, count: counts[d] || 0, color: decadeColors[i] }));
  }, [scatterPoints]);

  const similarPairs = [
    { a: { title: 'Motion Picture Soundtrack', artist: 'Radiohead' }, b: { title: 'The Night', artist: 'Zola Jesus' }, score: 0.96 },
    { a: { title: 'Holocene', artist: 'Bon Iver' }, b: { title: 'Skinny Love', artist: 'Bon Iver' }, score: 0.94 },
    { a: { title: 'Teardrop', artist: 'Massive Attack' }, b: { title: 'Unfinished Sympathy', artist: 'Massive Attack' }, score: 0.93 },
    { a: { title: 'Xtal', artist: 'Aphex Twin' }, b: { title: 'Lichen', artist: 'Aphex Twin' }, score: 0.91 },
    { a: { title: 'River Man', artist: 'Nick Drake' }, b: { title: 'Cello Song', artist: 'Nick Drake' }, score: 0.89 },
  ];

  const differentPairs = [
    { a: { title: 'Enter Sandman', artist: 'Metallica' }, b: { title: 'Clair de Lune', artist: 'Debussy' }, score: 0.05 },
    { a: { title: 'Windowlicker', artist: 'Aphex Twin' }, b: { title: 'So What', artist: 'Miles Davis' }, score: 0.07 },
    { a: { title: 'SICKO MODE', artist: 'Travis Scott' }, b: { title: 'Four Seasons', artist: 'Vivaldi' }, score: 0.09 },
    { a: { title: 'Acid Rain', artist: 'Chance the Rapper' }, b: { title: 'Gymnopédie No.1', artist: 'Satie' }, score: 0.11 },
    { a: { title: 'Paranoid Android', artist: 'Radiohead' }, b: { title: 'Kind of Blue', artist: 'Miles Davis' }, score: 0.13 },
  ];

  const c = {
    bg: isDark ? '#0d0d10' : '#f5f4f8',
    surface: isDark ? '#17171b' : '#ffffff',
    surface2: isDark ? '#1e1e23' : '#f0eff5',
    border: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)',
    text: isDark ? '#f0f0f5' : '#18181c',
    textMuted: isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)',
    textSubtle: isDark ? 'rgba(255,255,255,0.22)' : 'rgba(0,0,0,0.22)',
    grid: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
    accent: 'oklch(60% 0.18 270)',
    accentBg: isDark ? 'oklch(60% 0.18 270 / 0.12)' : 'oklch(60% 0.18 270 / 0.1)',
    scrollbar: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)',
  };

  const tabStyle = (active) => ({
    padding: '5px 12px', borderRadius: '7px', fontSize: '12px',
    fontFamily: 'Inter, sans-serif', fontWeight: active ? '600' : '400',
    border: 'none', cursor: 'pointer', transition: 'all 0.15s',
    background: active
      ? (isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)')
      : 'transparent',
    color: active ? c.text : c.textMuted,
    boxShadow: active ? (isDark ? 'inset 0 1px 0 rgba(255,255,255,0.05)' : 'inset 0 -1px 0 rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.08)') : 'none',
  });

  const colorOptions = [
    { id: 'genre', label: t.byGenre },
    { id: 'decade', label: t.byDecade },
    { id: 'artist', label: t.byArtist },
  ];

  const ScatterPlot2D = () => {
    const W = 520, H = 380;
    const PAD = 20;

    const handleMouseMove = (e) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const mx = (e.clientX - rect.left) / rect.width;
      const my = (e.clientY - rect.top) / rect.height;
      let closest = null, minDist = Infinity;
      scatterPoints.forEach(pt => {
        const dx = pt.x - mx, dy = pt.y - my;
        const d = Math.sqrt(dx*dx + dy*dy);
        if (d < minDist) { minDist = d; closest = pt; }
      });
      if (minDist < 0.04) {
        setTooltip({ pt: closest, mx: e.clientX - rect.left, my: e.clientY - rect.top });
      } else {
        setTooltip(null);
      }
    };

    return (
      <div style={{ position: 'relative', width: '100%' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          style={{ width: '100%', height: 'auto', cursor: 'crosshair' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
        >
          {/* Grid */}
          {[0.2, 0.4, 0.6, 0.8].map(v => (
            <g key={v}>
              <line x1={W*v} y1={PAD} x2={W*v} y2={H-PAD} stroke={c.grid} strokeWidth="1" />
              <line x1={PAD} y1={H*v} x2={W-PAD} y2={H*v} stroke={c.grid} strokeWidth="1" />
            </g>
          ))}
          {/* Axis labels */}
          <text x={W/2} y={H-4} textAnchor="middle" fontSize="10" fill={c.textSubtle} fontFamily="DM Mono, monospace">PCA 1</text>
          <text x={8} y={H/2} textAnchor="middle" fontSize="10" fill={c.textSubtle} fontFamily="DM Mono, monospace" transform={`rotate(-90, 8, ${H/2})`}>PCA 2</text>

          {/* Points */}
          {scatterPoints.map(pt => {
            const px = PAD + pt.x * (W - 2*PAD);
            const py = PAD + pt.y * (H - 2*PAD);
            const col = getPointColor(pt);
            const isHovered = tooltip?.pt?.id === pt.id;
            return (
              <circle
                key={pt.id}
                cx={px} cy={py}
                r={isHovered ? 6 : 3.5}
                fill={col}
                fillOpacity={isHovered ? 1 : 0.75}
                stroke={isHovered ? 'white' : col}
                strokeWidth={isHovered ? 1.5 : 0}
                style={{ transition: 'r 0.1s, fill-opacity 0.1s' }}
              />
            );
          })}
        </svg>

        {tooltip && (
          <div style={{
            position: 'absolute', left: tooltip.mx + 12, top: tooltip.my - 8,
            background: isDark ? '#1e1e26' : '#ffffff',
            border: `1px solid ${c.border}`,
            borderRadius: '10px', padding: '8px 12px',
            boxShadow: isDark ? '0 4px 16px rgba(0,0,0,0.5)' : '0 4px 16px rgba(0,0,0,0.12)',
            pointerEvents: 'none', zIndex: 10, minWidth: '140px',
          }}>
            <div style={{ fontSize: '12px', fontWeight: '600', color: c.text, fontFamily: 'Inter, sans-serif' }}>{tooltip.pt.title}</div>
            <div style={{ fontSize: '11px', color: c.textMuted, fontFamily: 'Inter, sans-serif', marginTop: '1px' }}>{tooltip.pt.artist}</div>
            <div style={{ display: 'flex', gap: '6px', marginTop: '6px', flexWrap: 'wrap' }}>
              <span style={{ padding: '1px 6px', borderRadius: '10px', fontSize: '10px', fontFamily: 'DM Mono, monospace',
                background: `${getPointColor(tooltip.pt).replace(')', '/0.15)')}`,
                color: getPointColor(tooltip.pt), border: `1px solid ${getPointColor(tooltip.pt).replace(')', '/0.25)')}` }}>
                {tooltip.pt.genre}
              </span>
              <span style={{ padding: '1px 6px', borderRadius: '10px', fontSize: '10px', fontFamily: 'DM Mono, monospace', color: c.textSubtle, border: `1px solid ${c.border}` }}>
                {tooltip.pt.year}
              </span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const ScatterPlot3D = () => {
    const W = 520, H = 380;
    const cx = W/2, cy = H/2;
    const scale = 160;
    const rotX = 0.4, rotY = 0.3;

    const project = (x, y, z) => {
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;
      const y1 = y * cosX - z1 * sinX;
      return { px: cx + x1 * scale, py: cy + y1 * scale, depth: z1 };
    };

    const pts3d = React.useMemo(() => {
      const seed = (n) => { let x = Math.sin(n) * 10000; return x - Math.floor(x); };
      return scatterPoints.map((pt, i) => ({
        ...pt,
        x3: (pt.x - 0.5) * 2,
        y3: (pt.y - 0.5) * 2,
        z3: (seed(i * 9.1) - 0.5) * 2,
      }));
    }, [scatterPoints]);

    const projected = pts3d.map(pt => ({
      ...pt,
      ...project(pt.x3, pt.y3, pt.z3),
    })).sort((a, b) => a.depth - b.depth);

    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        {/* Grid lines */}
        {[-1, -0.5, 0, 0.5, 1].map(v => {
          const a = project(v, -1, -1), b = project(v, -1, 1);
          const c2 = project(-1, v, -1), d = project(1, v, -1);
          return (
            <g key={v}>
              <line x1={a.px} y1={a.py} x2={b.px} y2={b.py} stroke={c.grid} strokeWidth="0.5" />
              <line x1={c2.px} y1={c2.py} x2={d.px} y2={d.py} stroke={c.grid} strokeWidth="0.5" />
            </g>
          );
        })}
        {projected.map(pt => (
          <circle key={pt.id}
            cx={pt.px} cy={pt.py}
            r={3 + pt.depth * 0.4}
            fill={getPointColor(pt)}
            fillOpacity={0.6 + pt.depth * 0.1}
          />
        ))}
        <text x={W/2} y={H-6} textAnchor="middle" fontSize="10" fill={c.textSubtle} fontFamily="DM Mono, monospace">3D PCA · interactive in full version</text>
      </svg>
    );
  };

  const GenreChart = () => {
    const total = genreCounts.reduce((s, g) => s + g.count, 0);
    const maxCount = Math.max(...genreCounts.map(g => g.count));
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {genreCounts.map((g, i) => (
          <div key={g.name} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '72px', fontSize: '12px', fontFamily: 'Inter, sans-serif', color: c.textMuted, textAlign: 'right', flexShrink: 0 }}>{g.name}</div>
            <div style={{ flex: 1, height: '22px', background: isDark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)', borderRadius: '5px', overflow: 'hidden',
              boxShadow: isDark?'inset 0 1px 2px rgba(0,0,0,0.3)':'inset 0 1px 2px rgba(0,0,0,0.08)' }}>
              <div style={{ height: '100%', width: `${(g.count / maxCount) * 100}%`, borderRadius: '5px',
                background: `linear-gradient(90deg, ${g.color}, ${g.color.replace('0.18', '0.12')})`,
                boxShadow: `0 0 6px ${g.color.replace(')','/0.3)')}`,
                transition: 'width 0.5s ease' }} />
            </div>
            <div style={{ width: '28px', fontSize: '12px', fontFamily: 'DM Mono, monospace', color: c.textSubtle }}>{g.count}</div>
          </div>
        ))}
      </div>
    );
  };

  const YearChart = () => {
    const maxCount = Math.max(...decadeCounts.map(d => d.count));
    return (
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '6px', height: '120px', paddingBottom: '20px', position: 'relative' }}>
        {decadeCounts.map((d, i) => (
          <div key={d.name} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', height: '100%', justifyContent: 'flex-end' }}>
            <div style={{ fontSize: '10px', fontFamily: 'DM Mono, monospace', color: c.textSubtle }}>{d.count}</div>
            <div style={{ width: '100%', borderRadius: '4px 4px 0 0',
              height: `${(d.count / maxCount) * 76}px`,
              background: `linear-gradient(0deg, ${d.color}, ${d.color.replace('0.18','0.1')})`,
              boxShadow: `0 0 8px ${d.color.replace(')','/0.25)')}`,
              transition: 'height 0.5s ease',
            }} />
            <div style={{ position: 'absolute', bottom: 0, fontSize: '10px', fontFamily: 'DM Mono, monospace', color: c.textSubtle, left: `${(i / decadeCounts.length) * 100}%`, width: `${100/decadeCounts.length}%`, textAlign: 'center' }}>{d.name}</div>
          </div>
        ))}
      </div>
    );
  };

  const DurationChart = () => {
    const buckets = [
      { label: '<2m', count: 8 }, { label: '2–3m', count: 32 }, { label: '3–4m', count: 67 },
      { label: '4–5m', count: 48 }, { label: '5–7m', count: 19 }, { label: '7m+', count: 6 },
    ];
    const maxCount = Math.max(...buckets.map(b => b.count));
    return (
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px', height: '120px', paddingBottom: '20px', position: 'relative' }}>
        {buckets.map((b, i) => (
          <div key={b.label} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'flex-end', gap: '4px' }}>
            <div style={{ fontSize: '10px', fontFamily: 'DM Mono, monospace', color: c.textSubtle }}>{b.count}</div>
            <div style={{ width: '100%', borderRadius: '4px 4px 0 0',
              height: `${(b.count / maxCount) * 76}px`,
              background: `linear-gradient(0deg, oklch(60% 0.18 ${200 + i*15}), oklch(55% 0.12 ${200 + i*15}))`,
              boxShadow: `0 0 6px oklch(60% 0.18 ${200 + i*15} / 0.25)`,
            }} />
            <div style={{ position: 'absolute', bottom: 0, fontSize: '9px', fontFamily: 'DM Mono, monospace', color: c.textSubtle, left: `${(i / buckets.length) * 100}%`, width: `${100/buckets.length}%`, textAlign: 'center' }}>{b.label}</div>
          </div>
        ))}
      </div>
    );
  };

  const PairCard = ({ pair, type }) => {
    const hueA = (pair.a.title.charCodeAt(0) * 37) % 360;
    const hueB = (pair.b.title.charCodeAt(0) * 37) % 360;
    const isSim = type === 'similar';
    return (
      <div style={{ padding: '10px 12px', borderRadius: '10px', background: c.surface, border: `1px solid ${c.border}`, display: 'flex', alignItems: 'center', gap: '10px',
        boxShadow: isDark ? '0 1px 4px rgba(0,0,0,0.3)' : '0 1px 4px rgba(0,0,0,0.06)' }}>
        <div style={{ display: 'flex', gap: '-6px', flexShrink: 0, position: 'relative', width: '58px' }}>
          <div style={{ width: '32px', height: '32px', borderRadius: '7px', zIndex: 2, position: 'relative',
            background: `linear-gradient(135deg, oklch(38% 0.12 ${hueA}), oklch(52% 0.18 ${(hueA+40)%360}))`,
            boxShadow: isDark ? 'inset 0 1px 0 rgba(255,255,255,0.1), 1px 0 0 rgba(0,0,0,0.3)' : 'inset 0 1px 0 rgba(255,255,255,0.5)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '10px', fontWeight: '700', color: 'rgba(255,255,255,0.6)', fontFamily: 'DM Mono, monospace' }}>
            {pair.a.title.slice(0,2).toUpperCase()}
          </div>
          <div style={{ width: '32px', height: '32px', borderRadius: '7px', position: 'absolute', left: '22px', zIndex: 1,
            background: `linear-gradient(135deg, oklch(38% 0.12 ${hueB}), oklch(52% 0.18 ${(hueB+40)%360}))`,
            boxShadow: isDark ? 'inset 0 1px 0 rgba(255,255,255,0.1)' : 'inset 0 1px 0 rgba(255,255,255,0.5)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '10px', fontWeight: '700', color: 'rgba(255,255,255,0.6)', fontFamily: 'DM Mono, monospace' }}>
            {pair.b.title.slice(0,2).toUpperCase()}
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: '12px', fontWeight: '600', color: c.text, fontFamily: 'Inter, sans-serif', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {pair.a.title} — {pair.b.title}
          </div>
          <div style={{ fontSize: '11px', color: c.textMuted, fontFamily: 'Inter, sans-serif', marginTop: '1px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {pair.a.artist} · {pair.b.artist}
          </div>
        </div>
        <div style={{ flexShrink: 0 }}>
          <span style={{ padding: '2px 8px', borderRadius: '12px', fontSize: '11px', fontFamily: 'DM Mono, monospace', fontWeight: '600',
            background: isSim ? 'oklch(62% 0.16 140 / 0.12)' : 'oklch(60% 0.18 310 / 0.12)',
            color: isSim ? 'oklch(62% 0.16 140)' : 'oklch(60% 0.18 310)',
            border: `1px solid ${isSim ? 'oklch(62% 0.16 140 / 0.25)' : 'oklch(60% 0.18 310 / 0.25)'}`,
          }}>
            {isSim ? `${Math.round(pair.score*100)}%` : `Δ ${pair.score.toFixed(2)}`}
          </span>
        </div>
      </div>
    );
  };

  const statCards = [
    { value: '180', label: t.totalTracks },
    { value: '42', label: t.totalArtists },
    { value: '31', label: t.totalAlbums },
    { value: '6', label: lang==='ru'?'жанров':'genres' },
  ];

  return (
    <div style={{ display: 'flex', flex: 1, flexDirection: 'column', height: '100vh', background: c.bg, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '18px 20px 14px', borderBottom: `1px solid ${c.border}`,
        background: isDark ? 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%)' : 'linear-gradient(180deg, rgba(255,255,255,0.8) 0%, transparent 100%)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif', letterSpacing: '-0.3px' }}>{t.title}</div>
            <div style={{ fontSize: '12px', color: c.textMuted, fontFamily: 'Inter, sans-serif', marginTop: '2px' }}>{t.subtitle}</div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {statCards.map(sc => (
              <div key={sc.label} style={{ textAlign: 'center', padding: '6px 12px', borderRadius: '10px',
                background: isDark?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.04)',
                border: `1px solid ${c.border}`,
                boxShadow: isDark ? 'inset 0 1px 0 rgba(255,255,255,0.03)' : 'inset 0 -1px 0 rgba(0,0,0,0.04)' }}>
                <div style={{ fontSize: '18px', fontWeight: '700', color: c.accent, fontFamily: 'DM Mono, monospace', letterSpacing: '-1px' }}>{sc.value}</div>
                <div style={{ fontSize: '10px', color: c.textSubtle, fontFamily: 'Inter, sans-serif', marginTop: '1px' }}>{sc.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left: scatter plot */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', borderRight: `1px solid ${c.border}` }}>
          <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '10px', borderBottom: `1px solid ${c.border}`, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', gap: '3px', background: isDark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)', padding: '3px', borderRadius: '8px' }}>
              <button style={tabStyle(scatterTab==='2d')} onClick={()=>setScatterTab('2d')}>{t.scatter2d}</button>
              <button style={tabStyle(scatterTab==='3d')} onClick={()=>setScatterTab('3d')}>{t.scatter3d}</button>
            </div>
            <div style={{ display: 'flex', gap: '3px', background: isDark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)', padding: '3px', borderRadius: '8px' }}>
              <button style={tabStyle(embedType==='sound')} onClick={()=>setEmbedType('sound')}>{t.soundEmbed}</button>
              <button style={tabStyle(embedType==='text')} onClick={()=>setEmbedType('text')}>{t.textEmbed}</button>
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace', color: c.textSubtle, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t.colorBy}</span>
              {colorOptions.map(o => (
                <button key={o.id} style={{ padding: '4px 9px', borderRadius: '6px', fontSize: '11px', fontFamily: 'Inter, sans-serif',
                  border: `1px solid ${colorBy===o.id ? 'oklch(60% 0.18 270 / 0.3)' : c.border}`,
                  background: colorBy===o.id ? c.accentBg : 'transparent',
                  color: colorBy===o.id ? c.accent : c.textMuted, cursor: 'pointer', fontWeight: colorBy===o.id?'600':'400' }}
                  onClick={()=>setColorBy(o.id)}>{o.label}</button>
              ))}
            </div>
          </div>

          <div style={{ flex: 1, overflow: 'hidden', position: 'relative', padding: '12px' }}>
            <div style={{ width: '100%', height: '100%', borderRadius: '10px', overflow: 'hidden',
              background: isDark?'rgba(255,255,255,0.02)':'rgba(0,0,0,0.02)',
              border: `1px solid ${c.border}` }}>
              {scatterTab === '2d' ? <ScatterPlot2D /> : <ScatterPlot3D />}
            </div>
          </div>

          {/* Legend */}
          <div style={{ padding: '8px 16px 12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {(colorBy === 'genre' ? genreCounts.map((g,i)=>({name:g.name, color:g.color})) : decadeCounts.map((d,i)=>({name:d.name, color:d.color}))).map(item => (
              <div key={item.name} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: item.color, flexShrink: 0,
                  boxShadow: `0 0 4px ${item.color.replace(')','/0.6)')}` }} />
                <span style={{ fontSize: '11px', fontFamily: 'Inter, sans-serif', color: c.textMuted }}>{item.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right panel */}
        <div style={{ width: '360px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Mini charts */}
          <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}` }}>
            <div style={{ display: 'flex', gap: '3px', marginBottom: '12px', background: isDark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)', padding: '3px', borderRadius: '8px', width: 'fit-content' }}>
              {[{id:'genres',label:t.genres},{id:'years',label:t.years},{id:'duration',label:t.duration}].map(tab => (
                <button key={tab.id} style={tabStyle(statsTab===tab.id)} onClick={()=>setStatsTab(tab.id)}>{tab.label}</button>
              ))}
            </div>
            {statsTab === 'genres' && <GenreChart />}
            {statsTab === 'years' && <YearChart />}
            {statsTab === 'duration' && <DurationChart />}
          </div>

          {/* Pairs */}
          <div style={{ flex: 1, overflowY: 'auto', scrollbarWidth: 'thin', scrollbarColor: `${c.scrollbar} transparent` }}>
            <div style={{ padding: '12px 14px 8px' }}>
              <div style={{ fontSize: '12px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'oklch(62% 0.16 140)', boxShadow: '0 0 4px oklch(62% 0.16 140 / 0.6)' }} />
                {t.similar}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                {similarPairs.map((pair, i) => <PairCard key={i} pair={pair} type="similar" />)}
              </div>
            </div>
            <div style={{ padding: '12px 14px 16px' }}>
              <div style={{ fontSize: '12px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'oklch(60% 0.18 310)', boxShadow: '0 0 4px oklch(60% 0.18 310 / 0.6)' }} />
                {t.different}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                {differentPairs.map((pair, i) => <PairCard key={i} pair={pair} type="different" />)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

Object.assign(window, { StatsSection });
