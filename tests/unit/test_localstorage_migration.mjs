// Phase D — unit test for runCollectionLocalStorageMigration_d.
//
// Extracts the migration helper straight out of frontend/index.html (single
// source of truth) and runs it against an in-memory localStorage stub. Run:
//
//   node tests/unit/test_localstorage_migration.mjs
//
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../../frontend/index.html', import.meta.url), 'utf8');

// ── Extract the function body via brace counting ──
const start = html.indexOf('function runCollectionLocalStorageMigration_d');
if (start < 0) { console.error('FAIL: migration fn not found in index.html'); process.exit(1); }
let i = html.indexOf('{', start), depth = 0, end = -1;
for (let j = i; j < html.length; j++) {
  if (html[j] === '{') depth++;
  else if (html[j] === '}') { depth--; if (depth === 0) { end = j + 1; break; } }
}
const fnSrc = html.slice(start, end);

// ── In-memory localStorage stub ──
function makeLS(initial) {
  const store = new Map(Object.entries(initial));
  return {
    get length() { return store.size; },
    key(n) { return Array.from(store.keys())[n] ?? null; },
    getItem(k) { return store.has(k) ? store.get(k) : null; },
    setItem(k, v) { store.set(k, String(v)); },
    removeItem(k) { store.delete(k); },
    _dump() { return Object.fromEntries(store); },
  };
}

// eslint-disable-next-line no-new-func
const runMigration = new Function('localStorage', `${fnSrc}\nreturn runCollectionLocalStorageMigration_d;`);

let failures = 0;
function check(name, cond) {
  if (cond) { console.log(`  ok   ${name}`); }
  else { console.error(`  FAIL ${name}`); failures++; }
}

// ── Scenario ──
const USER = 'user-A';
const ls = makeLS({
  'musix_token': 'tok',
  'active_collection': 'acct_user-A',
  'musix_chat_acct_user-A': '[{"id":1}]',         // ours (collection-suffixed)
  'musix_chat_acct_user-B': '[{"id":99}]',        // foreign account → drop
  'musix_chat_default': '[{"id":2}]',             // legacy pre-collection → claim if target empty
  'recentSearches:acct_user-A': '["wonderwall"]', // ours
  'recentSearches:acct_user-B': '["foreign"]',    // foreign → drop
  'chatHistory:track:T1': '[{"role":"user"}]',    // unsuffixed → add :userId
  'chatHistory:track:T2:user-A': '[{"x":1}]',     // already migrated → untouched
});

const fn = runMigration(ls);
fn(USER);
const after = ls._dump();

check('active_collection removed', !('active_collection' in after));
check('musix_chat_acct_user-A → musix_chat_user-A', after['musix_chat_user-A'] === '[{"id":1}]');
check('musix_chat_acct_user-A old key gone', !('musix_chat_acct_user-A' in after));
check('foreign musix_chat_acct_user-B dropped', !('musix_chat_acct_user-B' in after));
check('musix_chat_default dropped after claim attempt', !('musix_chat_default' in after));
check('recentSearches:acct_user-A → recentSearches:user-A', after['recentSearches:user-A'] === '["wonderwall"]');
check('foreign recentSearches:acct_user-B dropped', !('recentSearches:acct_user-B' in after));
check('chatHistory:track:T1 → :user-A', after['chatHistory:track:T1:user-A'] === '[{"role":"user"}]');
check('chatHistory:track:T1 old key gone', !('chatHistory:track:T1' in after));
check('already-migrated chatHistory:track:T2:user-A untouched', after['chatHistory:track:T2:user-A'] === '[{"x":1}]');
check('migration marker set', after['musix_localstorage_migration_d'] === 'done');
check('musix_token untouched', after['musix_token'] === 'tok');

// ── Idempotency: second run is a no-op ──
const before2 = JSON.stringify(ls._dump());
fn(USER);
check('idempotent (2nd run no-op)', JSON.stringify(ls._dump()) === before2);

// ── No-userId guard ──
const ls2 = makeLS({ 'active_collection': 'x' });
runMigration(ls2)('');
check('no-op when userId empty', ls2.getItem('active_collection') === 'x' && ls2.getItem('musix_localstorage_migration_d') === null);

console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL PASS');
process.exit(failures ? 1 : 0);
