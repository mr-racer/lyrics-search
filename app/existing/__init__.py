"""Existing user classes - wrappers for backward compatibility."""

from .folder_processor import FileProcessor
from .qdrant_db import LyricsDB

__all__ = ["FileProcessor", "LyricsDB"]
