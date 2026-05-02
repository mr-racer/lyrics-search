
// Chat/Search Section — split screen: chat left, results right
const ChatSection = ({ theme, lang }) => {
  const isDark = theme === 'dark';

  const T = {
    ru: {
      title: 'Поиск по коллекции',
      subtitle: 'Спроси что угодно о своей музыке',
      placeholder: 'Найди что-то похожее на Tool — Lateralus по звуку...',
      send: 'Отправить',
      searchMode: 'Режим поиска',
      modeText: 'Текст',
      modeSound: 'Звук',
      modeHybrid: 'Гибрид',
      filters: 'Фильтры',
      artist: 'Исполнитель',
      album: 'Альбом',
      yearRange: 'Диапазон лет',
      genre: 'Жанр',
      duration: 'Длительность',
      results: 'Результаты',
      upload: 'Загрузить трек',
      uploadHint: 'mp3, flac, m4a',
      noResults: 'Здесь появятся результаты поиска',
      noResultsHint: 'Задай вопрос или загрузи трек',
      msgPlaceholder1: 'Найди треки, похожие на эту мелодию — что-то медитативное, с гитарой',
      msgPlaceholder2: 'В текстах каких песен упоминается море или одиночество?',
      msgPlaceholder3: 'Что это за жанр? Найди похожее в моей библиотеке',
      similarity: 'схожесть',
      soundMatch: 'звук',
      lyricsMatch: 'текст',
      showCovers: 'Обложки',
      clearFilters: 'Сбросить',
      similarTo: 'Похожие на',
    },
    en: {
      title: 'Search Collection',
      subtitle: 'Ask anything about your music',
      placeholder: 'Find something similar to Tool — Lateralus by sound...',
      send: 'Send',
      searchMode: 'Search mode',
      modeText: 'Lyrics',
      modeSound: 'Sound',
      modeHybrid: 'Hybrid',
      filters: 'Filters',
      artist: 'Artist',
      album: 'Album',
      yearRange: 'Year range',
      genre: 'Genre',
      duration: 'Duration',
      results: 'Results',
      upload: 'Upload track',
      uploadHint: 'mp3, flac, m4a',
      noResults: 'Results will appear here',
      noResultsHint: 'Ask a question or upload a track',
      msgPlaceholder1: 'Find tracks similar to this melody — meditative, with guitar',
      msgPlaceholder2: 'Which songs mention the sea or loneliness in their lyrics?',
      msgPlaceholder3: 'What genre is this? Find similar in my library',
      similarity: 'similarity',
      soundMatch: 'sound',
      lyricsMatch: 'lyrics',
      showCovers: 'Covers',
      clearFilters: 'Clear',
      similarTo: 'Similar to',
    }
  };
  const t = T[lang] || T.ru;

  const [messages, setMessages] = React.useState([
    {
      role: 'assistant',
      text: lang === 'ru'
        ? 'Привет! Я помогу найти нужную музыку в твоей коллекции. Можешь описать настроение, загрузить трек для поиска похожих, или спросить по тексту песни.'
        : 'Hi! I can help you find music in your collection. Describe a mood, upload a track to find similar ones, or search by lyrics.',
      time: '14:32',
    },
    {
      role: 'user',
      text: lang === 'ru' ? 'Найди что-то медитативное с гитарой, похожее по звуку на Pink Floyd' : 'Find something meditative with guitar, similar to Pink Floyd',
      time: '14:33',
    },
    {
      role: 'assistant',
      text: lang === 'ru'
        ? 'Нашёл 5 треков в твоей коллекции с похожим звуковым профилем — плавные гитарные текстуры, умеренный темп, медитативная атмосфера:'
        : 'Found 5 tracks in your collection with a similar sound profile — smooth guitar textures, moderate tempo, meditative atmosphere:',
      time: '14:33',
      results: [
        { title: 'Breathe', artist: 'Pink Floyd', album: 'The Dark Side of the Moon', year: 1973, genre: 'Rock', duration: '2:43', score: 0.94, scoreType: 'sound' },
        { title: 'Shine On You Crazy Diamond', artist: 'Pink Floyd', album: 'Wish You Were Here', year: 1975, genre: 'Rock', duration: '13:32', score: 0.91, scoreType: 'sound' },
        { title: 'Hey You', artist: 'Pink Floyd', album: 'The Wall', year: 1979, genre: 'Rock', duration: '4:40', score: 0.87, scoreType: 'sound' },
      ],
    },
  ]);

  const [inputText, setInputText] = React.useState('');
  const [searchMode, setSearchMode] = React.useState('hybrid');
  const [showCovers, setShowCovers] = React.useState(true);
  const [showFilters, setShowFilters] = React.useState(false);
  const [filters, setFilters] = React.useState({ artist: '', album: '', genre: '', yearFrom: '1960', yearTo: '2024' });
  const [selectedResult, setSelectedResult] = React.useState(null);
  const chatEndRef = React.useRef(null);

  const c = {
    bg: isDark ? '#0d0d10' : '#f5f4f8',
    surface: isDark ? '#17171b' : '#ffffff',
    surface2: isDark ? '#1e1e23' : '#f0eff5',
    border: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)',
    text: isDark ? '#f0f0f5' : '#18181c',
    textMuted: isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)',
    textSubtle: isDark ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.25)',
    accent: 'oklch(60% 0.18 270)',
    accentBg: isDark ? 'oklch(60% 0.18 270 / 0.12)' : 'oklch(60% 0.18 270 / 0.1)',
    accentMuted: 'oklch(60% 0.18 310)',
    userBubble: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))',
    aiBubble: isDark ? '#1e1e23' : '#f0eff5',
    scrollbar: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)',
    inputBg: isDark ? '#1a1a1f' : '#ffffff',
    shadow: isDark ? '0 2px 12px rgba(0,0,0,0.5)' : '0 2px 12px rgba(0,0,0,0.08)',
  };

  const sendMessage = () => {
    if (!inputText.trim()) return;
    const newMsg = { role: 'user', text: inputText, time: new Date().toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'}) };
    setMessages(prev => [...prev, newMsg]);
    setInputText('');
    setTimeout(() => {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: lang === 'ru' ? 'Ищу по твоей коллекции...' : 'Searching your collection...',
        time: new Date().toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'}),
        loading: true,
      }]);
    }, 400);
  };

  const modes = [
    { id: 'text', label: t.modeText },
    { id: 'sound', label: t.modeSound },
    { id: 'hybrid', label: t.modeHybrid },
  ];

  const genreOptions = lang === 'ru'
    ? ['Все', 'Rock', 'Electronic', 'Jazz', 'Classical', 'Hip-Hop', 'Pop', 'Metal']
    : ['All', 'Rock', 'Electronic', 'Jazz', 'Classical', 'Hip-Hop', 'Pop', 'Metal'];

  const AlbumCover = ({ title, artist, size = 44 }) => {
    const hue = (title.charCodeAt(0) * 37 + artist.charCodeAt(0) * 17) % 360;
    return (
      <div style={{
        width: size, height: size, borderRadius: '8px', flexShrink: 0,
        background: `linear-gradient(135deg, oklch(40% 0.12 ${hue}), oklch(55% 0.18 ${(hue+40)%360}))`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: isDark ? 'inset 0 1px 0 rgba(255,255,255,0.1), 0 2px 6px rgba(0,0,0,0.4)' : 'inset 0 1px 0 rgba(255,255,255,0.5), 0 2px 6px rgba(0,0,0,0.1)',
        fontSize: '14px', fontWeight: '700', color: 'rgba(255,255,255,0.7)',
        fontFamily: 'DM Mono, monospace', letterSpacing: '-1px',
      }}>
        {title.slice(0,2).toUpperCase()}
      </div>
    );
  };

  const ScoreBadge = ({ score, type }) => (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '4px',
      padding: '2px 7px', borderRadius: '20px', fontSize: '11px',
      fontFamily: 'DM Mono, monospace', fontWeight: '500',
      background: type === 'sound' ? 'oklch(60% 0.18 310 / 0.15)' : 'oklch(60% 0.18 270 / 0.15)',
      color: type === 'sound' ? 'oklch(65% 0.18 310)' : 'oklch(65% 0.18 270)',
      border: `1px solid ${type === 'sound' ? 'oklch(60% 0.18 310 / 0.2)' : 'oklch(60% 0.18 270 / 0.2)'}`,
    }}>
      <span style={{ width: '6px', height: '6px', borderRadius: '50%',
        background: type === 'sound' ? 'oklch(60% 0.18 310)' : 'oklch(60% 0.18 270)' }}></span>
      {Math.round(score * 100)}% {type === 'sound' ? t.soundMatch : t.lyricsMatch}
    </div>
  );

  const ResultCard = ({ result, compact }) => (
    <div
      onClick={() => setSelectedResult(result)}
      style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        padding: compact ? '8px 10px' : '10px 12px',
        borderRadius: '10px', cursor: 'pointer',
        background: selectedResult === result
          ? (isDark ? 'oklch(60% 0.18 270 / 0.1)' : 'oklch(60% 0.18 270 / 0.07)')
          : 'transparent',
        border: `1px solid ${selectedResult === result ? 'oklch(60% 0.18 270 / 0.2)' : 'transparent'}`,
        transition: 'all 0.15s',
      }}
      onMouseEnter={e => { e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)'; }}
      onMouseLeave={e => { e.currentTarget.style.background = selectedResult === result ? (isDark?'oklch(60% 0.18 270/0.1)':'oklch(60% 0.18 270/0.07)') : 'transparent'; }}
    >
      {showCovers && <AlbumCover title={result.title} artist={result.artist} size={compact ? 36 : 44} />}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: '13px', fontWeight: '600', color: c.text, fontFamily: 'Inter, sans-serif',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{result.title}</div>
        <div style={{ fontSize: '12px', color: c.textMuted, fontFamily: 'Inter, sans-serif', marginTop: '1px' }}>
          {result.artist} · {result.year}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px', flexShrink: 0 }}>
        <ScoreBadge score={result.score} type={result.scoreType} />
        <span style={{ fontSize: '11px', color: c.textSubtle, fontFamily: 'DM Mono, monospace' }}>{result.duration}</span>
      </div>
    </div>
  );

  const containerStyle = {
    display: 'flex', flex: 1, height: '100vh', overflow: 'hidden', background: c.bg,
  };

  const chatPanelStyle = {
    width: '420px', minWidth: '360px', display: 'flex', flexDirection: 'column',
    borderRight: `1px solid ${c.border}`,
    background: isDark ? '#111115' : '#fafaf8',
  };

  const resultsPanelStyle = {
    flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
  };

  const headerStyle = {
    padding: '20px 20px 16px',
    borderBottom: `1px solid ${c.border}`,
    background: isDark
      ? 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%)'
      : 'linear-gradient(180deg, rgba(255,255,255,0.8) 0%, transparent 100%)',
  };

  const messagesStyle = {
    flex: 1, overflowY: 'auto', padding: '16px',
    display: 'flex', flexDirection: 'column', gap: '12px',
    scrollbarWidth: 'thin', scrollbarColor: `${c.scrollbar} transparent`,
  };

  const inputAreaStyle = {
    padding: '12px 16px 16px',
    borderTop: `1px solid ${c.border}`,
    background: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.6)',
    backdropFilter: 'blur(8px)',
  };

  const modeBarStyle = {
    display: 'flex', gap: '4px', marginBottom: '10px',
  };

  const modeBtnStyle = (id) => ({
    flex: 1, padding: '6px', borderRadius: '8px', border: 'none',
    fontSize: '12px', fontFamily: 'Inter, sans-serif', fontWeight: searchMode === id ? '600' : '400',
    cursor: 'pointer', transition: 'all 0.15s',
    background: searchMode === id
      ? 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))'
      : (isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)'),
    color: searchMode === id ? 'white' : c.textMuted,
    boxShadow: searchMode === id
      ? '0 2px 8px oklch(60% 0.18 270 / 0.3), inset 0 1px 0 rgba(255,255,255,0.15)'
      : (isDark ? 'inset 0 1px 0 rgba(255,255,255,0.03)' : 'inset 0 -1px 0 rgba(0,0,0,0.05)'),
  });

  const inputRowStyle = {
    display: 'flex', gap: '8px', alignItems: 'flex-end',
  };

  const textareaStyle = {
    flex: 1, padding: '10px 12px', borderRadius: '10px', border: `1px solid ${c.border}`,
    background: c.inputBg, color: c.text, resize: 'none', fontFamily: 'Inter, sans-serif',
    fontSize: '13px', lineHeight: '1.5', outline: 'none', minHeight: '40px', maxHeight: '120px',
    boxShadow: isDark
      ? 'inset 0 1px 3px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(255,255,255,0.03)'
      : 'inset 0 1px 3px rgba(0,0,0,0.06)',
  };

  const sendBtnStyle = {
    width: '40px', height: '40px', borderRadius: '10px', border: 'none',
    background: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))',
    color: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
    boxShadow: '0 2px 8px oklch(60% 0.18 270 / 0.35), inset 0 1px 0 rgba(255,255,255,0.2)',
    transition: 'all 0.15s', flexShrink: 0,
  };

  const uploadBtnStyle = {
    width: '40px', height: '40px', borderRadius: '10px', border: `1px dashed ${c.border}`,
    background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)',
    color: c.textMuted, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
    transition: 'all 0.15s', flexShrink: 0,
  };

  const filtersBtnStyle = {
    display: 'flex', alignItems: 'center', gap: '6px',
    padding: '5px 10px', borderRadius: '8px', border: `1px solid ${c.border}`,
    background: showFilters ? c.accentBg : (isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)'),
    color: showFilters ? c.accent : c.textMuted,
    fontSize: '12px', fontFamily: 'Inter, sans-serif', cursor: 'pointer', transition: 'all 0.15s',
  };

  const renderMessage = (msg, i) => {
    const isUser = msg.role === 'user';
    return (
      <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', gap: '4px' }}>
        <div style={{
          maxWidth: '88%', padding: '10px 13px', borderRadius: isUser ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
          background: isUser ? c.userBubble : c.aiBubble,
          color: isUser ? 'white' : c.text,
          fontSize: '13px', lineHeight: '1.6', fontFamily: 'Inter, sans-serif',
          boxShadow: isDark ? '0 1px 4px rgba(0,0,0,0.3)' : '0 1px 4px rgba(0,0,0,0.06)',
        }}>
          {msg.loading ? (
            <div style={{ display: 'flex', gap: '4px', alignItems: 'center', padding: '2px 0' }}>
              {[0,1,2].map(j => (
                <div key={j} style={{
                  width: '6px', height: '6px', borderRadius: '50%',
                  background: c.textMuted,
                  animation: `pulse 1.2s ease-in-out ${j*0.2}s infinite`,
                }}></div>
              ))}
            </div>
          ) : msg.text}
        </div>
        {msg.results && (
          <div style={{
            maxWidth: '92%', borderRadius: '10px', overflow: 'hidden',
            border: `1px solid ${c.border}`, background: c.surface,
            boxShadow: isDark ? '0 2px 8px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.06)',
          }}>
            {msg.results.map((r, ri) => <ResultCard key={ri} result={r} compact />)}
          </div>
        )}
        <span style={{ fontSize: '10px', color: c.textSubtle, fontFamily: 'DM Mono, monospace' }}>{msg.time}</span>
      </div>
    );
  };

  const FiltersPanel = () => (
    <div style={{
      margin: '0 16px 8px', padding: '14px', borderRadius: '12px',
      background: c.surface2, border: `1px solid ${c.border}`,
      display: 'flex', flexDirection: 'column', gap: '10px',
    }}>
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        {[
          { key: 'artist', label: t.artist },
          { key: 'album', label: t.album },
        ].map(f => (
          <div key={f.key} style={{ flex: 1, minWidth: '120px' }}>
            <div style={{ fontSize: '10px', fontFamily: 'DM Mono,monospace', color: c.textSubtle, marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{f.label}</div>
            <input
              value={filters[f.key]}
              onChange={e => setFilters(p => ({...p, [f.key]: e.target.value}))}
              placeholder={`— ${lang==='ru'?'любой':'any'} —`}
              style={{
                width: '100%', padding: '6px 10px', borderRadius: '8px',
                background: c.inputBg, border: `1px solid ${c.border}`,
                color: c.text, fontSize: '12px', fontFamily: 'Inter, sans-serif',
                outline: 'none', boxSizing: 'border-box',
                boxShadow: isDark ? 'inset 0 1px 2px rgba(0,0,0,0.3)' : 'inset 0 1px 2px rgba(0,0,0,0.04)',
              }}
            />
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <div style={{ flex: 2, minWidth: '160px' }}>
          <div style={{ fontSize: '10px', fontFamily: 'DM Mono,monospace', color: c.textSubtle, marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t.yearRange}</div>
          <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
            <input value={filters.yearFrom} onChange={e => setFilters(p=>({...p,yearFrom:e.target.value}))} style={{ width: '64px', padding: '6px 8px', borderRadius: '8px', background: c.inputBg, border: `1px solid ${c.border}`, color: c.text, fontSize: '12px', fontFamily: 'DM Mono,monospace', outline: 'none', boxShadow: isDark?'inset 0 1px 2px rgba(0,0,0,0.3)':'inset 0 1px 2px rgba(0,0,0,0.04)' }} />
            <span style={{ color: c.textSubtle, fontSize: '12px' }}>—</span>
            <input value={filters.yearTo} onChange={e => setFilters(p=>({...p,yearTo:e.target.value}))} style={{ width: '64px', padding: '6px 8px', borderRadius: '8px', background: c.inputBg, border: `1px solid ${c.border}`, color: c.text, fontSize: '12px', fontFamily: 'DM Mono,monospace', outline: 'none', boxShadow: isDark?'inset 0 1px 2px rgba(0,0,0,0.3)':'inset 0 1px 2px rgba(0,0,0,0.04)' }} />
          </div>
        </div>
        <div style={{ flex: 2, minWidth: '140px' }}>
          <div style={{ fontSize: '10px', fontFamily: 'DM Mono,monospace', color: c.textSubtle, marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t.genre}</div>
          <select value={filters.genre} onChange={e => setFilters(p=>({...p,genre:e.target.value}))} style={{ width: '100%', padding: '6px 10px', borderRadius: '8px', background: c.inputBg, border: `1px solid ${c.border}`, color: c.text, fontSize: '12px', fontFamily: 'Inter, sans-serif', outline: 'none', cursor: 'pointer', boxShadow: isDark?'inset 0 1px 2px rgba(0,0,0,0.3)':'inset 0 1px 2px rgba(0,0,0,0.04)' }}>
            {genreOptions.map(g => <option key={g} value={g === 'Все' || g === 'All' ? '' : g}>{g}</option>)}
          </select>
        </div>
        <button onClick={() => setFilters({ artist:'', album:'', genre:'', yearFrom:'1960', yearTo:'2024' })} style={{ padding: '6px 12px', borderRadius: '8px', border: `1px solid ${c.border}`, background: 'transparent', color: c.textMuted, fontSize: '12px', fontFamily: 'Inter, sans-serif', cursor: 'pointer' }}>
          {t.clearFilters}
        </button>
      </div>
    </div>
  );

  // Detailed result panel
  const DetailPanel = ({ result }) => (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '20px' }}>
      <div style={{ width: '120px', height: '120px', borderRadius: '20px',
        background: `linear-gradient(135deg, oklch(40% 0.12 ${(result.title.charCodeAt(0)*37)%360}), oklch(55% 0.18 ${(result.title.charCodeAt(0)*37+40)%360}))`,
        boxShadow: isDark ? 'inset 0 2px 0 rgba(255,255,255,0.15), 0 12px 40px rgba(0,0,0,0.5)' : 'inset 0 2px 0 rgba(255,255,255,0.6), 0 12px 40px rgba(0,0,0,0.12)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '32px', fontWeight: '700', color: 'rgba(255,255,255,0.6)',
        fontFamily: 'DM Mono, monospace',
      }}>
        {result.title.slice(0,2).toUpperCase()}
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '22px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif', letterSpacing: '-0.5px' }}>{result.title}</div>
        <div style={{ fontSize: '15px', color: c.textMuted, fontFamily: 'Inter, sans-serif', marginTop: '4px' }}>{result.artist}</div>
        <div style={{ fontSize: '13px', color: c.textSubtle, fontFamily: 'Inter, sans-serif', marginTop: '2px' }}>{result.album} · {result.year}</div>
      </div>
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
        {[
          { label: result.genre, color: c.accent },
          { label: result.duration, color: c.accentMuted },
          { label: `${Math.round(result.score*100)}% match`, color: 'oklch(60% 0.15 140)' },
        ].map((tag, i) => (
          <span key={i} style={{ padding: '4px 12px', borderRadius: '20px', fontSize: '12px',
            fontFamily: 'DM Mono, monospace',
            background: `${tag.color.replace('oklch','oklch').replace(')','/0.12)')}`,
            color: tag.color, border: `1px solid ${tag.color.replace(')','/0.2)')}` }}>
            {tag.label}
          </span>
        ))}
      </div>
      <button style={{
        padding: '10px 24px', borderRadius: '10px', border: 'none',
        background: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 300))',
        color: 'white', fontSize: '14px', fontFamily: 'Inter, sans-serif', fontWeight: '600',
        cursor: 'pointer',
        boxShadow: '0 4px 12px oklch(60% 0.18 270 / 0.3), inset 0 1px 0 rgba(255,255,255,0.2)',
      }}>
        {lang === 'ru' ? '→ Найти похожие' : '→ Find similar'}
      </button>
    </div>
  );

  return (
    <div style={containerStyle}>
      <style>{`@keyframes pulse { 0%,100%{opacity:0.3;transform:scale(0.8)} 50%{opacity:1;transform:scale(1)} }`}</style>

      {/* Chat Panel */}
      <div style={chatPanelStyle}>
        <div style={headerStyle}>
          <div style={{ fontSize: '16px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif', letterSpacing: '-0.3px' }}>{t.title}</div>
          <div style={{ fontSize: '12px', color: c.textMuted, fontFamily: 'Inter, sans-serif', marginTop: '2px' }}>{t.subtitle}</div>
        </div>

        <div style={messagesStyle}>
          {messages.map(renderMessage)}
          <div ref={chatEndRef} />
        </div>

        {showFilters && <FiltersPanel />}

        <div style={inputAreaStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <div style={modeBarStyle}>
              {modes.map(m => (
                <button key={m.id} style={modeBtnStyle(m.id)} onClick={() => setSearchMode(m.id)}>{m.label}</button>
              ))}
            </div>
            <button style={filtersBtnStyle} onClick={() => setShowFilters(p => !p)}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
              {t.filters}
            </button>
            <button
              onClick={() => setShowCovers(p => !p)}
              style={{ ...filtersBtnStyle, background: showCovers ? c.accentBg : (isDark?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.03)'), color: showCovers ? c.accent : c.textMuted }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              {t.showCovers}
            </button>
          </div>
          <div style={inputRowStyle}>
            <button style={uploadBtnStyle} title={t.uploadHint}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </button>
            <textarea
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }}}
              placeholder={t.placeholder}
              rows={1}
              style={textareaStyle}
            />
            <button style={sendBtnStyle} onClick={sendMessage}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
            </button>
          </div>
        </div>
      </div>

      {/* Results Panel */}
      <div style={resultsPanelStyle}>
        <div style={{ ...headerStyle, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: c.text, fontFamily: 'Inter, sans-serif', letterSpacing: '-0.3px' }}>{t.results}</div>
            <div style={{ fontSize: '12px', color: c.textMuted, fontFamily: 'Inter, sans-serif', marginTop: '2px' }}>
              {lang === 'ru' ? '3 трека · режим: ' : '3 tracks · mode: '}<span style={{ color: c.accent, fontFamily: 'DM Mono, monospace' }}>{searchMode}</span>
            </div>
          </div>
        </div>

        {selectedResult ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '8px 16px', borderBottom: `1px solid ${c.border}` }}>
              <button onClick={() => setSelectedResult(null)} style={{ background: 'none', border: 'none', color: c.textMuted, cursor: 'pointer', fontSize: '13px', fontFamily: 'Inter, sans-serif', display: 'flex', alignItems: 'center', gap: '4px' }}>
                ← {lang === 'ru' ? 'назад' : 'back'}
              </button>
            </div>
            <DetailPanel result={selectedResult} />
          </div>
        ) : (
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px', scrollbarWidth: 'thin', scrollbarColor: `${c.scrollbar} transparent` }}>
            {messages.filter(m => m.results).flatMap(m => m.results).length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {messages.filter(m => m.results).flatMap(m => m.results).map((r, i) => (
                  <ResultCard key={i} result={r} />
                ))}
              </div>
            ) : (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 40px', gap: '12px' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '14px',
                  background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={c.textSubtle} strokeWidth="1.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
                </div>
                <div style={{ fontSize: '14px', fontWeight: '600', color: c.textMuted, fontFamily: 'Inter, sans-serif' }}>{t.noResults}</div>
                <div style={{ fontSize: '12px', color: c.textSubtle, fontFamily: 'Inter, sans-serif' }}>{t.noResultsHint}</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

Object.assign(window, { ChatSection });
