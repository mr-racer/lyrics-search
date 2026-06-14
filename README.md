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

## 🚀 Quick Start (Anaconda — run it on your own machine)

This branch runs MusiX **directly on your computer** with **Anaconda** — a free,
beginner-friendly Python toolkit. You don't need to know how to code: open one
window, then copy each command, paste it, and press **Enter**.

### Step 0 — Install the tools (one time only)

1. **Anaconda** — download from <https://www.anaconda.com/download> and install
   it with the default options.
2. **Docker Desktop** — download from
   <https://www.docker.com/products/docker-desktop/> and install it. MusiX keeps
   its search index in a small database called **Qdrant**, and the simplest way
   to run Qdrant is through Docker. Start Docker Desktop once and leave it
   running in the background (its whale icon should say "running").

### Step 1 — Open the Anaconda Prompt

Click **Start**, type **Anaconda Prompt**, and open it. A black window appears —
this is where you paste the commands below. *(On macOS/Linux, use your normal
Terminal instead.)*

### Step 2 — Go to the project folder

Replace the path with wherever you unzipped MusiX, then press **Enter**:

```bash
cd C:\Users\YourName\Desktop\lyrics-search
```

### Step 3 — Create the `musix` environment (one time only)

This makes a clean, isolated space for MusiX so it never clashes with other
software on your computer:

```bash
conda create -n musix python=3.11 -y
conda activate musix
```

Your prompt should now start with `(musix)`. **Tip:** every time you open a new
Anaconda Prompt, run `conda activate musix` again before using MusiX.

### Step 4 — Install MusiX and everything it needs (one time only)

```bash
pip install -r requirements.txt
```

This downloads MusiX plus its AI models — several gigabytes, so the first run
takes a while. Grab a coffee. ☕

> 💡 A recent **NVIDIA GPU** is recommended (the install grabs GPU-accelerated
> AI libraries). It will still install without one, but searching will be
> slower.

### Step 5 — Start the Qdrant database

Make sure **Docker Desktop is running**, then:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

You only create it once. After that you can start/stop it from the Docker
Desktop window, or with `docker start qdrant` / `docker stop qdrant`.

### Step 6 — Start MusiX

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --log-config logging.conf
```

Keep this window open — it **is** the running app. To stop MusiX later, click
the window and press **Ctrl + C**.

### Step 7 — Open it in your browser

Go to **<http://localhost:8000>**, point MusiX at your music folder, and start
searching. 🎉

> ℹ️ Don't double-click `index.html` — MusiX is a web app, so the page only
> works while the server from Step 6 is running and you open it through
> `localhost:8000`.

> 🔎 *Optional:* web search for missing lyrics uses **SearXNG**. To enable it,
> run `docker-compose up -d` (this branch's compose file starts SearXNG).
> MusiX works fine without it.

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
