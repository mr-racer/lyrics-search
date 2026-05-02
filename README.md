# 🎵 MusiX — Find Music by Meaning, Not by Name

> **Stop guessing search queries. Start describing what you remember.**

You remember the vibe, a lyric fragment, or that one line about a car — but not the song title. MusiX finds it anyway.

Drop your local music library into MusiX, and it builds a **semantic index** of every track: lyrics, mood, genre, metadata. Then you ask in plain language — and get ranked results with matching lyric snippets and album art.

---

## ✨ What can it do?

### 🔍 Semantic Lyrics Search

Describe a song in your own words, and MusiX understands the *meaning*, not just keywords:

> _"Song where the singer talks about driving a Mercedes at night"_
> _"Что-то про дождь и городские огни, женский голос"_

No more "song that goes dah-dah-dah-dah" on YouTube. MusiX uses vector embeddings to match your description against the actual semantic content of every lyric in your library.

### 🎧 Search by Sound & Mood

Not just lyrics — search by **acoustic characteristics**. Describe the sound, energy, or atmosphere:

> _"Aggressive trap beat with 808 bass"_
> _"Медленная акустическая баллада с фортепиано"_

Under the hood, MusiX uses **CLAP** (Contrastive Language-Audio Pretraining) to bridge text descriptions and audio embeddings — so you can find songs by how they *sound*, not just what they *say*.

### 💬 AI Chat Assistant

Don't feel like crafting the perfect query? Chat with MusiX naturally. The built-in AI assistant:

- Understands conversational context across multiple rounds
- Translates your casual descriptions into precise search queries
- Returns ranked results with confidence scores
- Remembers your conversation history per collection

### 📚 Multi-Library Collections

Index multiple folders into separate named collections — switch between them instantly:

- Your personal MP3 archive
- A band's complete discography
- A curated playlist of FLAC rips

Each collection gets its own vector index, chat history, and search scope.

### 🖼️ Album Art

MusiX extracts cover art directly from your audio files and displays it alongside search results — no missing covers, no broken links.

---

## 🚀 Quick Start

```bash
# 1. Start Qdrant (vector database)
docker-compose up -d

# 2. Install dependencies
pip install -e .

# 3. Run the server
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000 --log-config logging.conf
```

Open `http://localhost:8000` → point MusiX at your music folder → start searching.

---

## 🧠 How it works (briefly)

1. **Index** — MusiX scans your folder, extracts metadata (title, artist, album, genre, duration) and lyrics from file tags. Missing lyrics are auto-fetched from online sources.
2. **Embed** — Every song's lyrics are converted into a dense vector (Sentence Transformers). Audio characteristics are embedded separately (CLAP).
3. **Search** — Your query is embedded the same way. Cosine similarity ranks the closest matches. Hybrid mode fuses both signals.
4. **Display** — Results appear with lyric snippets, confidence scores, album art, and metadata filters.

---

## 📋 Roadmap

| Feature | Status |
|---------|--------|
| Semantic lyrics search | ✅ Done |
| Audio/mood search (CLAP) | ✅ Done |
| AI chat assistant | ✅ Done |
| Album art extraction | ✅ Done |
| Multi-collection support | ✅ Done |
| Smart recommendations | 🚧 In progress |
| Audio waveform visualization | 📋 Planned |
| Mobile-responsive UI | 📋 Planned |

---

## ⚙️ Configuration

Set via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Vector database endpoint |
| `TEXT_MODEL` | `jinaai/jina-embeddings-v2-small-en` | Lyrics embedding model |
| `AUDIO_MODEL` | `laion/clap-htsat-base` | Audio embedding model |
| `LLM_BASE_URL` | — | OpenAI-compatible LLM endpoint (LM Studio, Ollama, etc.) |
| `MUSIC_FOLDER` | — | Default music library path |

---

## 📜 License

MIT — use it, fork it, remix it.
