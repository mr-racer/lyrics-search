
// Sidebar navigation component
const Sidebar = ({ activeSection, onNavigate, theme, onThemeToggle, lang, onLangToggle }) => {
  const t = lang === 'ru' ? {
    search: 'Поиск',
    recommend: 'Рекомендации',
    stats: 'Статистика',
    collection: 'Коллекция',
    settings: 'Настройки',
  } : {
    search: 'Search',
    recommend: 'Recommendations',
    stats: 'Analytics',
    collection: 'Collection',
    settings: 'Settings',
  };

  const navItems = [
    { id: 'chat', label: t.search, icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
        <path d="M11 8v6M8 11h6" strokeWidth="1.5"/>
      </svg>
    )},
    { id: 'recommend', label: t.recommend, icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
      </svg>
    )},
    { id: 'stats', label: t.stats, icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
      </svg>
    )},
  ];

  const isDark = theme === 'dark';

  const sidebarStyle = {
    width: '220px',
    minWidth: '220px',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: isDark
      ? 'linear-gradient(180deg, #111114 0%, #0d0d10 100%)'
      : 'linear-gradient(180deg, #f0eff5 0%, #e8e7ef 100%)',
    borderRight: isDark ? '1px solid rgba(255,255,255,0.06)' : '1px solid rgba(0,0,0,0.08)',
    boxShadow: isDark
      ? 'inset -1px 0 0 rgba(255,255,255,0.03), 4px 0 24px rgba(0,0,0,0.4)'
      : 'inset -1px 0 0 rgba(0,0,0,0.04), 4px 0 16px rgba(0,0,0,0.08)',
    padding: '0',
    zIndex: 10,
    flexShrink: 0,
  };

  const logoStyle = {
    padding: '28px 24px 24px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  };

  const logoIconStyle = {
    width: '34px',
    height: '34px',
    borderRadius: '10px',
    background: 'linear-gradient(135deg, oklch(60% 0.18 270), oklch(60% 0.18 310))',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 2px 8px oklch(60% 0.18 270 / 0.5), inset 0 1px 0 rgba(255,255,255,0.2)',
    flexShrink: 0,
  };

  const logoTextStyle = {
    fontSize: '18px',
    fontWeight: '700',
    letterSpacing: '-0.5px',
    color: isDark ? '#f0f0f5' : '#18181c',
    fontFamily: 'Inter, sans-serif',
  };

  const navStyle = {
    flex: 1,
    padding: '8px 12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  };

  const getNavItemStyle = (id) => {
    const isActive = activeSection === id;
    return {
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      padding: '10px 12px',
      borderRadius: '10px',
      cursor: 'pointer',
      fontSize: '14px',
      fontWeight: isActive ? '600' : '400',
      fontFamily: 'Inter, sans-serif',
      color: isActive
        ? 'oklch(60% 0.18 270)'
        : (isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.45)'),
      background: isActive
        ? (isDark ? 'oklch(60% 0.18 270 / 0.12)' : 'oklch(60% 0.18 270 / 0.1)')
        : 'transparent',
      boxShadow: isActive
        ? (isDark ? 'inset 0 1px 0 rgba(255,255,255,0.05), 0 1px 3px rgba(0,0,0,0.3)' : 'inset 0 1px 0 rgba(255,255,255,0.8), 0 1px 3px rgba(0,0,0,0.08)')
        : 'none',
      transition: 'all 0.15s ease',
      userSelect: 'none',
      letterSpacing: '-0.1px',
    };
  };

  const bottomStyle = {
    padding: '16px 12px 20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    borderTop: isDark ? '1px solid rgba(255,255,255,0.06)' : '1px solid rgba(0,0,0,0.07)',
  };

  const toggleRowStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '6px 12px',
  };

  const toggleBtnStyle = (active) => ({
    padding: '4px 10px',
    borderRadius: '6px',
    fontSize: '12px',
    fontFamily: 'DM Mono, monospace',
    fontWeight: '500',
    cursor: 'pointer',
    border: 'none',
    background: active
      ? (isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)')
      : 'transparent',
    color: active
      ? (isDark ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.8)')
      : (isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)'),
    transition: 'all 0.15s',
    boxShadow: active
      ? (isDark ? 'inset 0 1px 0 rgba(255,255,255,0.05)' : 'inset 0 -1px 0 rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1)')
      : 'none',
  });

  const themeBtnStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '13px',
    fontFamily: 'Inter, sans-serif',
    color: isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)',
    background: 'transparent',
    border: 'none',
    transition: 'all 0.15s',
    width: '100%',
    textAlign: 'left',
  };

  return (
    <div style={sidebarStyle}>
      <div style={logoStyle}>
        <div style={logoIconStyle}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
            <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
          </svg>
        </div>
        <span style={logoTextStyle}>MusiX</span>
      </div>

      <nav style={navStyle}>
        {navItems.map(item => (
          <div
            key={item.id}
            style={getNavItemStyle(item.id)}
            onClick={() => onNavigate(item.id)}
            onMouseEnter={e => {
              if (activeSection !== item.id) {
                e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
                e.currentTarget.style.color = isDark ? 'rgba(255,255,255,0.75)' : 'rgba(0,0,0,0.7)';
              }
            }}
            onMouseLeave={e => {
              if (activeSection !== item.id) {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.45)';
              }
            }}
          >
            {item.icon}
            <span>{item.label}</span>
          </div>
        ))}
      </nav>

      <div style={bottomStyle}>
        <div style={toggleRowStyle}>
          <span style={{fontSize:'11px', fontFamily:'DM Mono,monospace', color: isDark?'rgba(255,255,255,0.25)':'rgba(0,0,0,0.25)', marginRight:'4px'}}>LANG</span>
          <button style={toggleBtnStyle(lang==='ru')} onClick={() => onLangToggle('ru')}>RU</button>
          <button style={toggleBtnStyle(lang==='en')} onClick={() => onLangToggle('en')}>EN</button>
        </div>
        <button style={themeBtnStyle} onClick={onThemeToggle}>
          {isDark ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
          )}
          {isDark ? (lang==='ru'?'Светлая тема':'Light mode') : (lang==='ru'?'Тёмная тема':'Dark mode')}
        </button>
      </div>
    </div>
  );
};

Object.assign(window, { Sidebar });
