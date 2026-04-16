# lyrics-search
RAG-powered semantic search over song lyrics — ask in plain English, find the exact track.

Builds a vector database from song lyrics (fetching missing lyrics automatically via LLM), then enables natural-language queries like "Kanye West song where he mentions Mercedes CL" — returning ranked matches with context.

# Features:
Auto-fetches missing lyrics for songs in your DB
* Embeds lyrics into a vector store for semantic search
* Natural language query interface via RAG pipeline
* Returns song matches with relevant lyric snippets
