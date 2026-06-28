# 🎵 MusiX — A Local Music Player That Actually Knows Your Taste

> **Your music. Your machine. An AI that listens with you — not instead of you.**

MusiX is a music player for **your own library** — the files you already own, on your
own computer. Point it at a folder of MP3/FLAC, and on top of a clean player it adds
an AI layer that learns what you love: it builds a portrait of your taste, keeps an
endless personal stream going, writes playlists from a one-line wish, tells you the
story behind every artist and song, and chats with you about whatever is playing.

No streaming catalog. No "songs you might also like" pulled from strangers. No
AI-generated tracks, no auto-remixes — **only the music you chose to add.**

---

## 🔒 Local-first, by design

This is the whole point, so it comes first:

- **Everything runs on your machine** — the player, the AI brain, the database. Nothing
  about your listening is uploaded to a MusiX cloud, because there isn't one.
- **You choose the library.** MusiX only ever sees the folder you point it at. It never
  reaches out to add tracks, suggest "official" versions, or slip in AI-made songs or
  remixes. What you put in is exactly what you get back.
- **Your taste data stays yours** — likes, play history, the taste portrait, and chat
  history live in a local SQLite file on your disk.
- **The AI can be fully local too.** MusiX talks to any OpenAI-compatible LLM endpoint,
  so you can run the language model on your own GPU with **LM Studio** or **Ollama** and
  keep the entire experience offline.

---

## ✨ What can it do?

### 🎧 A real player for your own files

The new heart of MusiX is the player itself: gapless playback of your local library,
album art pulled straight from your files, lyrics in view, a reactive equalizer, and an
"aurora" mode that paints the screen with the mood of the current track. Organize your
music into multiple libraries and switch between them instantly.

### 🌊 An endless personal stream

Hit play and never stop. MusiX keeps an infinite, personalized stream running — each next
track chosen from **your** library to match where your ears are right now. When it runs
out of fresh picks it gracefully loops back instead of dead-ending, so the music never
just stops.

### 🪞 Your musical taste, put into words

MusiX doesn't just recommend — it can **explain you to yourself**. From what you actually
play and like, it builds:

- **Taste islands** — the distinct clusters of sound you keep coming back to, each given
  a punchy name ("Late-Night Synthwave", not "Artist X").
- **A sound-axis profile** — where you sit on energy, vocals, spaciousness, brightness,
  acousticness, and how experimental you lean.
- **A listener portrait** — a few honest sentences about what you love, contradictions
  included.
- **A "wave" tagline** on the For-You screen that captures the mood you're in *today*,
  blending your lasting taste with what's been on rotation lately.

### 🪄 Playlists from a single wish

Describe what you want in plain language — _"energetic rock about love"_, _"a calm
late-night jazz set"_ — and MusiX builds a curated, well-ordered playlist **out of your
own library**, with a short reason next to each pick explaining why it made the cut. It
plans the search, pulls real tracks from your collection, and sequences them so the set
flows.

### 📖 Bios & facts about artists and songs

Every artist gets a researched one-paragraph biography; every song gets a set of curated
facts — the story, the context, the trivia. Tap into them while you listen, or browse a
stream of random facts about the music you own.

### 💬 Chat about the song that's playing

Open the chat drawer on any track and ask anything:

- _"What is this song actually about?"_
- _"Does this sample something?"_
- _"Why does the bridge change key here?"_

The assistant answers from the song's full lyrics, metadata, and facts — and only reaches
out to the web when the answer genuinely isn't in front of it (and tells you honestly when
it can't find something).

### 🔎 Pinpoint questions about the lyrics

See a line you don't get? Select it and ask. MusiX explains **that exact line** — the
reference, the wordplay, the meaning — in a couple of sentences, quoting the phrase back
to you instead of giving a generic summary.

---

## 🚀 Quick Start (Docker — the easy way)

Everything — the app, the search database (Qdrant), and the web-search helper
(SearXNG) — runs inside **Docker** containers. You don't install Python or any
AI models by hand; **one command** downloads and starts it all.

### Step 0 — Install Docker Desktop (one time only)

Download **Docker Desktop** from
<https://www.docker.com/products/docker-desktop/> and install it with the
default options. Start it once and wait until its whale icon says it's running.

> 💡 On Windows, Docker Desktop will offer to enable **WSL2** — say yes.

### Step 1 — Get the project

Download this branch as a ZIP from GitHub and unzip it (or `git clone` it if you
know Git). Remember where you put the folder.

### Step 2 — Open a terminal **inside** the project folder

- **Windows:** open the unzipped folder in File Explorer, click the address bar
  at the top, type `cmd`, and press **Enter**. A black window opens, already
  pointed at the folder.
- **macOS/Linux:** open Terminal and `cd` into the folder.

### Step 3 — (Optional) add your settings

MusiX works out of the box, but you can plug in an AI chat key and a security
secret. Make your own settings file:

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Then open `.env` in any text editor and fill in the blanks. You can skip this
and do it later.

### Step 4 — Start everything

```bash
docker compose up -d --build
```

The first run downloads several gigabytes (the app image + AI models), so it can
take a while. When it finishes, MusiX is running quietly in the background.

### Step 5 — Open it in your browser

Go to **<http://localhost:8000>**. The first-run wizard creates your owner
account → point MusiX at your music (upload or index) → start listening. 🎉

### Everyday commands

```bash
docker compose ps        # see what's running
docker compose logs -f   # watch the app's output (Ctrl+C just stops watching)
docker compose down      # stop everything
docker compose up -d     # start again later (no --build needed next time)
```

> **GPU note:** the `musix` service requests an NVIDIA GPU for faster AI. It
> needs Docker Desktop with the WSL2 backend **+** the NVIDIA Container Toolkit.
> **No GPU?** Open `docker-compose.yml`, delete the `deploy.resources` block
> under the `musix` service, and switch `requirements.txt` to the CPU torch
> wheels — then run Step 4 again.

<details>
<summary>Verifying the live stack (real Qdrant + real models)</summary>

Most tests mock Qdrant and stub the ML libraries, so `pytest` runs anywhere.
A small live-stack suite under `tests/docker/` instead exercises the real
thing — dense embeddings, a real Qdrant accepting and ranking them, and CLAP
actually loading. Run it inside the running container:

```bash
docker compose up -d
scripts/run_docker_tests.sh
```

First run downloads the embedding model (~1 GB) and the CLAP checkpoint
(~2.3 GB) into the mounted volumes; later runs reuse them.
</details>

<details>
<summary>Bare-metal (no Docker) — Windows dev</summary>

```bash
# 1. Start Qdrant + SearXNG
docker compose up -d qdrant searxng

# 2. Install dependencies
pip install -e .

# 3. Run the server (QDRANT_URL defaults to http://localhost:6333)
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000 --log-config logging.conf
```
</details>

---

## 🧠 How it works (briefly)

1. **Add your library** — MusiX scans your folder, reads tags (title, artist, album,
   genre, duration) and lyrics, and extracts cover art. Missing lyrics are filled in
   automatically.
2. **Learn the sound** — each track is analyzed for its acoustic character and lyrical
   content, so MusiX can tell what sits near what *inside your own collection*.
3. **Learn you** — your plays and likes shape a living taste profile: islands, sound
   axes, and a portrait, all recomputed as your taste drifts.
4. **Play it back** — the player streams your files, the stream keeps choosing what's
   next, the AI writes playlists and bios, and the chat answers questions — all from the
   music you own.

---

## 📋 Roadmap

| Feature | Status |
|---------|--------|
| Local music player (stream, aurora, EQ) | ✅ Done |
| Endless personalized stream | ✅ Done |
| Taste profile (islands, axes, portrait) | ✅ Done |
| AI playlist generation from a wish | ✅ Done |
| Artist bios & song facts | ✅ Done |
| Per-track chat + lyric line explanations | ✅ Done |
| Album art & multi-library support | ✅ Done |
| Audio waveform visualization | 📋 Planned |
| Mobile-responsive UI | 📋 Planned |

---

## ⚙️ Configuration

Set via a `.env` file (copy `.env.example` to `.env`) or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://qdrant:6333` | Local taste-index database endpoint (use `http://localhost:6333` for a bare-metal run) |
| `LLM_BASE_URL` | — | OpenAI-compatible LLM endpoint (LM Studio, Ollama, or a hosted API) |
| `LLM_MODEL` | — | LLM model name to use |
| `OPENAI_API_KEY` | — | API key for the LLM endpoint (any value for local servers) |
| `TEXT_MODEL` | `jinaai/jina-embeddings-v2-small-en` | Lyrics analysis model |
| `MUSIX_JWT_SECRET` | — | 32+ random chars for auth (`openssl rand -hex 32`) |
| `MUSIC_FOLDER` | — | Default music library path |

Without an LLM configured, the player and your library work fully — only the AI
features (taste portrait, playlists, chat, bios) stay disabled.

---

## 📜 License

Licensed under the **Apache License, Version 2.0**. You may use, modify, and
distribute MusiX, including for commercial purposes, provided you keep the
required attribution and `NOTICE`. The license also includes an explicit patent
grant. See the [LICENSE](LICENSE) file for the full terms.
