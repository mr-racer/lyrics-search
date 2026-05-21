"""Existing user classes - wrappers for backward compatibility."""

# LyricsDB is safe to import eagerly (no file_processor dependency).
from .qdrant_db import LyricsDB

# FileProcessor is imported lazily to avoid a circular import:
#   app.indexing.* → app.__init__ → app.existing → file_processor.main → file_processor.utils → app.indexing.*
def __getattr__(name: str):
    if name == "FileProcessor":
        from .folder_processor import FileProcessor  # noqa: PLC0415
        globals()["FileProcessor"] = FileProcessor
        return FileProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FileProcessor", "LyricsDB"]
