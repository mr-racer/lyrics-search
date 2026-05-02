
// Recommendations Section — AI assistant + playlist builder (2 modes)
const RecommendSection = ({ theme, lang }) => {
  const isDark = theme === 'dark';

  const T = {
    ru: {
      title: 'Рекомендации',
      subtitle: 'ИИ-ассистент подберёт музыку под твоё настроение',
      modeChat: 'Диалог с ИИ',
      modeBuilder: 'Конструктор',
      placeholder: 'Что сейчас слушаешь? Или опиши настроение...',
      send: 'Отправить',
      createPlaylist: 'Создать плейлист',
      playlistName: 'Название плейлиста',
      playlistNamePH: 'Вечерний джаз, Концентрация...',
      mood: 'Настроение',
      energy: 'Энергетика',
      era: 'Эпоха',
      genre: 'Жанр',
      trackCount: 'Кол-во треков',
      generate: 'Сгенерировать',
      playlistReady: 'Плейлист готов',
      export: 'Экспорт',
      tracksIn: 'треков',
      moods: ['Меланхолия', 'Радость', 'Концентрация', 'Расслабление', 'Драйв', 'Ностальгия'],
      eras: ['60-е', '70-е', '80-е', '90-е', '00-е', '10-е', '20-е'],
      genres: ['Rock', 'Electronic', 'Jazz', 'Classical', 'Hip-Hop', 'Pop', 'Metal', 'Folk'],
      aiGreeting: 'Привет! Я помогу подобрать музыку специально для тебя. Расскажи, что ты сейчас чувствуешь или чем занимаешься?',
      aiQ1: 'Отличный выбор! Тебе сейчас больше хочется энергичного или спокойного?',
      aiQ2: 'Понял. Предпочтёшь что-то из своей коллекции или хочешь выйти за её рамки?',
      playlistTitle: 'Вечерний поток',
    },
    en: {
      title: 'Recommendations',
      subtitle: 'AI assistant picks music for your mood',
      modeChat: 'AI Dialog',
      modeBuilder: 'Builder',
      placeholder: 'What are you listening to? Or describe your mood...',
      send: 'Send',
      createPlaylist: 'Create playlist',
      playlistName: 'Playlist name',
      playlistNamePH: 'Evening jazz, Focus...',
      mood: 'Mood',
      energy: 'Energy',
      era: 'Era',
      genre: 'Genre',
      trackCount: 'Track count',
      generate: 'Generate',
      playlistReady: 'Playlist ready',
      export: 'Export',
      tracksIn: 'tracks',
      moods: ['Melancholy', 'Joy', 'Focus', 'Chill', 'Drive', 'Nostalgia'],
      eras: ["'60s", "'70s", "'80s", "'90s", "'00s", "'10s", "'20s"],
      genres: ['Rock', 'Electronic', 'Jazz', 'Classical', 'Hip-Hop', 'Pop', 'Metal', 'Folk'],
      aiGreeting: 'Hey! I\'ll help you find the perfect music. Tell me, how are you feeling right now or what are you doing?',
      aiQ1: 'Great! Do you prefer something energetic or calm right now?',
      aiQ2: 'Got it. Would you like something from your library or are you open to new discoveries?',
      playlistTitle: 'Evening Flow',
    }
  };
  const t = T[lang] || T.ru;

  const [mode, setMode] = React.useState('chat');
  const [messages, setMessages] = React.useState([
    { role: 'assistant', text: t.aiGreeting, time: '14:40' },
    { role: 'user', text: lang === 'ru' ? 'Хочу что-то для вечерней прогулки, немного меланхоличное' : 'I want something for an evening walk, a bit melancholic', time: '14:41' },
    { role: 'assistant', text: t.aiQ1, time: '14:41' },
    { role: 'user', text: lang === 'ru' ? 'Спокойное, медленное' : 'Calm, slow', time: '14:42' },
    { role: 'assistant', text: t.aiQ2, time: '14:42' },
  ]);
  const [inputText, setInputText] = React.useState('');

  // Builder state
  const [builderState, setBuilderState] = React.useState({
    name: '',
    moods: [],
    energy: 40,
    eras: [],
    genres: [],
    count: 15,
  });
  const [generated, setGenerated] = React.useState(false);

  const samplePlaylist = [
    { title: 'Fake Plastic Trees', artist: 'Radiohead', duration: '4:50', score: 0.93 },
    { title: 'The Night', artist: 'Zola Jesus', duration: '3:42', score: 0.91 },
    { title: 'Motion Picture Soundtrack', artist: 'Radiohead', duration: '5:00', score: 0.88 },
    { title: 'Teardrop', artist: 'Massive Attack', duration: '5:30', score: 0.87 },
    { title: 'Holocene', artist: 'Bon Iver', duration: '5:37', score: 0.85 },
    { title: 'Skinny Love', artist: 'Bon Iver', duration: '3:58', score: 0.83 },
  ];

  const c = {
    bg: isDark ? '#0d0d10' : '#f5f4f8',
    surface: isDark ? '#17171b' : '#ffffff',
    surface2: isDark ? '#1e1e23' : '#f0eff5',
    border: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)',
    text: isDark ? '#f0f0f5' : '#18181c',
    textMuted: isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)',
    textSubtle: isDark ? 'rgba(255,255,255,0.22)' : 'rgba(0,0,0,0.22)',
    accent: 'oklch(60% 0.18 270)',
    accentBg: isDark ? 'oklch(60% 0.18 270 / 0.12)' : 'oklch(60% 0.18 270 / 0.1)',
    accent2: 'oklch(60% 0.18 310)',
    accent2Bg: isDark ? 'oklch(60% 0.18 310 / 0.12)' : 'oklch(60% 0.18 310 / 0.1)',
    userBubble: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))',
    aiBubble: isDark ? '#1e1e23' : '#f0eff5',
    inputBg: isDark ? '#1a1a1f' : '#ffffff',
    scrollbar: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)',
  };

  const sendMessage = () => {
    if (!inputText.trim()) return;
    setMessages(prev => [...prev, { role: 'user', text: inputText, time: new Date().toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'}) }]);
    setInputText('');
    setTimeout(() => {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: lang === 'ru'
          ? 'Отлично! На основе твоих ответов я составил плейлист из 12 треков — меланхоличный, медленный, из твоей коллекции. Хочешь увидеть?'
          : 'Great! Based on your answers I\'ve built a playlist of 12 tracks — melancholic, slow, from your library. Want to see it?',
        time: new Date().toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'}),
        hasPlaylist: true,
      }]);
    }, 600);
  };

  const toggleArr = (arr, val) =>
    arr.includes(val) ? arr.filter(x => x !== val) : [...arr, val];

  const chipStyle = (active) => ({
    padding: '5px 11px', borderRadius: '20px', fontSize: '12px',
    fontFamily: 'Inter, sans-serif', fontWeight: active ? '600' : '400',
    cursor: 'pointer', border: `1px solid ${active ? 'oklch(60% 0.18 270 / 0.4)' : c.border}`,
    background: active ? c.accentBg : (isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)'),
    color: active ? c.accent : c.textMuted,
    transition: 'all 0.15s', userSelect: 'none',
    boxShadow: active ? 'inset 0 1px 0 rgba(255,255,255,0.08)' : 'none',
  });

  const labelStyle = {
    fontSize: '11px', fontFamily: 'DM Mono, monospace', color: c.textSubtle,
    textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '8px',
  };

  const AlbumCover = ({ title, artist, size = 40 }) => {
    const hue = (title.charCodeAt(0) * 37 + artist.charCodeAt(0) * 17) % 360;
    return (
      <div style={{
        width: size, height: size, borderRadius: '8px', flexShrink: 0,
        background: `linear-gradient(135deg, oklch(38% 0.12 ${hue}), oklch(52% 0.18 ${(hue+40)%360}))`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: isDark ? 'inset 0 1px 0 rgba(255,255,255,0.1)' : 'inset 0 1px 0 rgba(255,255,255,0.5)',
        fontSize: '12px', fontWeight: '700', color: 'rgba(255,255,255,0.65)',
        fontFamily: 'DM Mono, monospace',
      }}>
        {title.slice(0,2).toUpperCase()}
      </div>
    );
  };

  const ChatMode = () => (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Chat */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: `1px solid ${c.border}`, minWidth: 0 }}>
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', scrollbarWidth: 'thin', scrollbarColor: `${c.scrollbar} transparent` }}>
          {messages.map((msg, i) => {
            const isUser = msg.role === 'user';
            return (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', gap: '4px' }}>
                {!isUser && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                    <div style={{ width: '20px', height: '20px', borderRadius: '6px',
                      background: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))',
                      display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 6v6l4 2"/></svg>
                    </div>
                    <span style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace', color: c.textSubtle }}>MusiX AI</span>
                  </div>
                )}
                <div style={{
                  maxWidth: '80%', padding: '10px 13px',
                  borderRadius: isUser ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                  background: isUser ? c.userBubble : c.aiBubble,
                  color: isUser ? 'white' : c.text,
                  fontSize: '13px', lineHeight: '1.6', fontFamily: 'Inter, sans-serif',
                  boxShadow: isDark ? '0 1px 4px rgba(0,0,0,0.3)' : '0 1px 4px rgba(0,0,0,0.06)',
                }}>
                  {msg.text}
                </div>
                {msg.hasPlaylist && (
                  <div style={{
                    maxWidth: '80%', padding: '10px 12px', borderRadius: '10px',
                    background: c.surface, border: `1px solid ${c.border}`,
                    display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer',
                    boxShadow: isDark ? '0 2px 8px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.06)',
                  }}>
                    <div style={{ width: '36px', height: '36px', borderRadius: '8px',
                      background: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 310))',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.2)' }}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
                    </div>
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: '600', color: c.text, fontFamily: 'Inter, sans-serif' }}>{t.playlistTitle}</div>
                      <div style={{ fontSize: '11px', color: c.textMuted, fontFamily: 'DM Mono, monospace' }}>12 {t.tracksIn}</div>
                    </div>
                    <div style={{ marginLeft: 'auto', color: c.accent, fontSize: '12px', fontFamily: 'Inter, sans-serif', fontWeight: '600' }}>
                      {lang === 'ru' ? 'Открыть →' : 'Open →'}
                    </div>
                  </div>
                )}
                <span style={{ fontSize: '10px', color: c.textSubtle, fontFamily: 'DM Mono, monospace' }}>{msg.time}</span>
              </div>
            );
          })}
        </div>
        <div style={{ padding: '12px 16px', borderTop: `1px solid ${c.border}`, background: isDark?'rgba(0,0,0,0.2)':'rgba(255,255,255,0.5)', backdropFilter: 'blur(8px)' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => { if(e.key==='Enter') sendMessage(); }}
              placeholder={t.placeholder}
              style={{
                flex: 1, padding: '10px 14px', borderRadius: '10px',
                background: c.inputBg, border: `1px solid ${c.border}`,
                color: c.text, fontSize: '13px', fontFamily: 'Inter, sans-serif',
                outline: 'none',
                boxShadow: isDark ? 'inset 0 1px 3px rgba(0,0,0,0.4)' : 'inset 0 1px 3px rgba(0,0,0,0.06)',
              }}
            />
            <button onClick={sendMessage} style={{
              width: '42px', height: '42px', borderRadius: '10px', border: 'none',
              background: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))',
              color: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
              boxShadow: '0 2px 8px oklch(60% 0.18 270 / 0.35), inset 0 1px 0 rgba(255,255,255,0.2)',
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
            </button>
          </div>
        </div>
      </div>

      {/* Quick playlist preview */}
      <div style={{ width: '280px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '16px 16px 10px', borderBottom: `1px solid ${c.border}` }}>
          <div style={{ fontSize: '13px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif' }}>
            {lang === 'ru' ? 'Последний плейлист' : 'Latest playlist'}
          </div>
          <div style={{ fontSize: '11px', color: c.textMuted, fontFamily: 'DM Mono, monospace', marginTop: '2px' }}>
            {t.playlistTitle} · 6 {t.tracksIn}
          </div>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px', scrollbarWidth: 'thin', scrollbarColor: `${c.scrollbar} transparent` }}>
          {samplePlaylist.map((track, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '7px 8px', borderRadius: '8px', transition: 'background 0.15s', cursor: 'pointer' }}
              onMouseEnter={e => e.currentTarget.style.background = isDark?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.03)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
              <span style={{ width: '16px', fontSize: '11px', fontFamily: 'DM Mono, monospace', color: c.textSubtle, textAlign: 'right', flexShrink: 0 }}>{i+1}</span>
              <AlbumCover title={track.title} artist={track.artist} size={34} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '12px', fontWeight: '500', color: c.text, fontFamily: 'Inter, sans-serif', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{track.title}</div>
                <div style={{ fontSize: '11px', color: c.textMuted, fontFamily: 'Inter, sans-serif' }}>{track.artist}</div>
              </div>
              <span style={{ fontSize: '11px', color: c.textSubtle, fontFamily: 'DM Mono, monospace', flexShrink: 0 }}>{track.duration}</span>
            </div>
          ))}
        </div>
        <div style={{ padding: '10px 12px', borderTop: `1px solid ${c.border}` }}>
          <button style={{
            width: '100%', padding: '9px', borderRadius: '10px', border: 'none',
            background: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))',
            color: 'white', fontSize: '13px', fontFamily: 'Inter, sans-serif', fontWeight: '600',
            cursor: 'pointer',
            boxShadow: '0 2px 8px oklch(60% 0.18 270 / 0.3), inset 0 1px 0 rgba(255,255,255,0.2)',
          }}>
            {t.export}
          </button>
        </div>
      </div>
    </div>
  );

  const BuilderMode = () => (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Controls */}
      <div style={{ width: '320px', display: 'flex', flexDirection: 'column', overflowY: 'auto', padding: '20px', gap: '20px', borderRight: `1px solid ${c.border}`, scrollbarWidth: 'thin', scrollbarColor: `${c.scrollbar} transparent` }}>
        {/* Playlist name */}
        <div>
          <div style={labelStyle}>{t.playlistName}</div>
          <input
            value={builderState.name}
            onChange={e => setBuilderState(p => ({...p, name: e.target.value}))}
            placeholder={t.playlistNamePH}
            style={{
              width: '100%', padding: '9px 12px', borderRadius: '10px',
              background: c.inputBg, border: `1px solid ${c.border}`,
              color: c.text, fontSize: '13px', fontFamily: 'Inter, sans-serif',
              outline: 'none', boxSizing: 'border-box',
              boxShadow: isDark ? 'inset 0 1px 3px rgba(0,0,0,0.4)' : 'inset 0 1px 3px rgba(0,0,0,0.06)',
            }}
          />
        </div>

        {/* Mood chips */}
        <div>
          <div style={labelStyle}>{t.mood}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {t.moods.map(m => (
              <span key={m} style={chipStyle(builderState.moods.includes(m))}
                onClick={() => setBuilderState(p => ({...p, moods: toggleArr(p.moods, m)}))}>
                {m}
              </span>
            ))}
          </div>
        </div>

        {/* Energy slider */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <div style={labelStyle}>{t.energy}</div>
            <span style={{ fontSize: '12px', fontFamily: 'DM Mono, monospace', color: c.accent }}>{builderState.energy}%</span>
          </div>
          <div style={{ position: 'relative', height: '20px', display: 'flex', alignItems: 'center' }}>
            <div style={{ position: 'absolute', width: '100%', height: '4px', borderRadius: '2px', background: isDark?'rgba(255,255,255,0.08)':'rgba(0,0,0,0.08)',
              boxShadow: isDark?'inset 0 1px 2px rgba(0,0,0,0.4)':'inset 0 1px 2px rgba(0,0,0,0.1)' }}>
              <div style={{ width: `${builderState.energy}%`, height: '100%', borderRadius: '2px',
                background: 'linear-gradient(90deg, oklch(60% 0.18 270), oklch(60% 0.18 310))',
                boxShadow: '0 0 6px oklch(60% 0.18 270 / 0.4)' }} />
            </div>
            <input type="range" min={0} max={100} value={builderState.energy}
              onChange={e => setBuilderState(p => ({...p, energy: +e.target.value}))}
              style={{ width: '100%', position: 'relative', opacity: 0, cursor: 'pointer', height: '20px', margin: 0 }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
            <span style={{ fontSize: '10px', fontFamily: 'DM Mono, monospace', color: c.textSubtle }}>
              {lang === 'ru' ? 'спокойно' : 'calm'}
            </span>
            <span style={{ fontSize: '10px', fontFamily: 'DM Mono, monospace', color: c.textSubtle }}>
              {lang === 'ru' ? 'энергично' : 'energetic'}
            </span>
          </div>
        </div>

        {/* Era */}
        <div>
          <div style={labelStyle}>{t.era}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
            {t.eras.map(e => (
              <span key={e} style={chipStyle(builderState.eras.includes(e))}
                onClick={() => setBuilderState(p => ({...p, eras: toggleArr(p.eras, e)}))}>
                {e}
              </span>
            ))}
          </div>
        </div>

        {/* Genre */}
        <div>
          <div style={labelStyle}>{t.genre}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
            {t.genres.map(g => (
              <span key={g} style={chipStyle(builderState.genres.includes(g))}
                onClick={() => setBuilderState(p => ({...p, genres: toggleArr(p.genres, g)}))}>
                {g}
              </span>
            ))}
          </div>
        </div>

        {/* Track count */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <div style={labelStyle}>{t.trackCount}</div>
            <span style={{ fontSize: '13px', fontFamily: 'DM Mono, monospace', color: c.accent, fontWeight: '600' }}>{builderState.count}</span>
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            {[5, 10, 15, 20, 25, 30].map(n => (
              <button key={n} onClick={() => setBuilderState(p => ({...p, count: n}))} style={{
                flex: 1, padding: '6px', borderRadius: '8px', border: `1px solid ${builderState.count===n ? 'oklch(60% 0.18 270 / 0.4)' : c.border}`,
                background: builderState.count===n ? c.accentBg : (isDark?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.03)'),
                color: builderState.count===n ? c.accent : c.textMuted,
                fontSize: '12px', fontFamily: 'DM Mono, monospace', cursor: 'pointer',
                fontWeight: builderState.count===n ? '700' : '400',
              }}>{n}</button>
            ))}
          </div>
        </div>

        <button onClick={() => setGenerated(true)} style={{
          padding: '12px', borderRadius: '12px', border: 'none',
          background: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))',
          color: 'white', fontSize: '14px', fontFamily: 'Inter, sans-serif', fontWeight: '600',
          cursor: 'pointer', marginTop: '4px',
          boxShadow: '0 4px 16px oklch(60% 0.18 270 / 0.35), inset 0 1px 0 rgba(255,255,255,0.2)',
          letterSpacing: '-0.2px',
        }}>
          ✦ {t.generate}
        </button>
      </div>

      {/* Generated playlist */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {generated ? (
          <>
            <div style={{ padding: '20px 20px 14px', borderBottom: `1px solid ${c.border}` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif', letterSpacing: '-0.3px' }}>
                    {builderState.name || t.playlistTitle}
                  </div>
                  <div style={{ fontSize: '12px', color: c.textMuted, fontFamily: 'DM Mono, monospace', marginTop: '2px' }}>
                    {builderState.count} {t.tracksIn}
                    {builderState.moods.length > 0 && ` · ${builderState.moods.join(', ')}`}
                  </div>
                </div>
                <button style={{
                  padding: '7px 14px', borderRadius: '8px', border: 'none',
                  background: c.accentBg, color: c.accent,
                  fontSize: '12px', fontFamily: 'Inter, sans-serif', fontWeight: '600',
                  cursor: 'pointer', border: `1px solid oklch(60% 0.18 270 / 0.2)`,
                }}>{t.export}</button>
              </div>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '8px 12px', scrollbarWidth: 'thin', scrollbarColor: `${c.scrollbar} transparent` }}>
              {samplePlaylist.map((track, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '9px 8px', borderRadius: '9px', cursor: 'pointer', transition: 'background 0.15s' }}
                  onMouseEnter={e => e.currentTarget.style.background = isDark?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.03)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                  <span style={{ width: '20px', fontSize: '12px', fontFamily: 'DM Mono, monospace', color: c.textSubtle, textAlign: 'right', flexShrink: 0 }}>{i+1}</span>
                  <AlbumCover title={track.title} artist={track.artist} size={38} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '13px', fontWeight: '600', color: c.text, fontFamily: 'Inter, sans-serif', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{track.title}</div>
                    <div style={{ fontSize: '12px', color: c.textMuted, fontFamily: 'Inter, sans-serif' }}>{track.artist}</div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
                    <span style={{ padding: '2px 7px', borderRadius: '12px', fontSize: '11px', fontFamily: 'DM Mono, monospace',
                      background: 'oklch(60% 0.18 270 / 0.1)', color: 'oklch(65% 0.18 270)',
                      border: '1px solid oklch(60% 0.18 270 / 0.2)' }}>
                      {Math.round(track.score * 100)}%
                    </span>
                    <span style={{ fontSize: '12px', color: c.textSubtle, fontFamily: 'DM Mono, monospace' }}>{track.duration}</span>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '14px', padding: '40px' }}>
            <div style={{ width: '56px', height: '56px', borderRadius: '16px',
              background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: isDark ? 'inset 0 1px 0 rgba(255,255,255,0.06)' : 'inset 0 -1px 0 rgba(0,0,0,0.06)' }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={c.textSubtle} strokeWidth="1.5" strokeLinecap="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
            </div>
            <div style={{ fontSize: '14px', fontWeight: '600', color: c.textMuted, fontFamily: 'Inter, sans-serif' }}>
              {lang === 'ru' ? 'Настрой параметры и нажми «Сгенерировать»' : 'Set parameters and click Generate'}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const headerStyle = {
    padding: '20px 20px 0',
    background: isDark
      ? 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%)'
      : 'linear-gradient(180deg, rgba(255,255,255,0.8) 0%, transparent 100%)',
    borderBottom: `1px solid ${c.border}`,
  };

  const tabStyle = (active) => ({
    padding: '10px 18px 12px', fontSize: '13px', fontFamily: 'Inter, sans-serif',
    fontWeight: active ? '600' : '400',
    color: active ? c.accent : c.textMuted,
    cursor: 'pointer', border: 'none', background: 'none',
    borderBottom: `2px solid ${active ? c.accent : 'transparent'}`,
    transition: 'all 0.15s', marginBottom: '-1px',
  });

  return (
    <div style={{ display: 'flex', flex: 1, flexDirection: 'column', height: '100vh', background: c.bg, overflow: 'hidden' }}>
      <div style={headerStyle}>
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '16px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif', letterSpacing: '-0.3px' }}>{t.title}</div>
          <div style={{ fontSize: '12px', color: c.textMuted, fontFamily: 'Inter, sans-serif', marginTop: '2px' }}>{t.subtitle}</div>
        </div>
        <div style={{ display: 'flex', gap: '0' }}>
          <button style={tabStyle(mode === 'chat')} onClick={() => setMode('chat')}>{t.modeChat}</button>
          <button style={tabStyle(mode === 'builder')} onClick={() => setMode('builder')}>{t.modeBuilder}</button>
        </div>
      </div>
      {mode === 'chat' ? <ChatMode /> : <BuilderMode />}
    </div>
  );
};

Object.assign(window, { RecommendSection });
