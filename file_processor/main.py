from .utils import fetch_lyrics_bulk, research_with_musixmatch

from pathlib import Path
import json
import re


class FileProcessor:
    """Collects music file metadata and fetches lyrics for tracks in a folder."""

    def __init__(self):
        self.metadata = []

    def process_folder(self, music_folder: str, better_lyrics_quality: bool = False):
        processed_files = fetch_lyrics_bulk(music_folder=music_folder, better_lyrics_quality=better_lyrics_quality)
        self.metadata = processed_files
        return processed_files

    def load(self, path: str):
        try:
            path_directory = Path(path)
            with open(str(path_directory), "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            print("Ошибка чтения результатов! Введен неверный путь. ", e)

    def to_dict(self) -> dict:
        if self.metadata:
            return self.metadata
        else:
            print("Не найдены данные для выгрузки")
        
    def to_json(self, directory: Path = Path.cwd() / "metadata.json"):
        if self.metadata:
            try:
                path_directory = Path(directory)
                with open(str(path_directory), "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                    print(f"Файл с метаданными был успешно сохранен в {str(path_directory)}")
            except Exception as e:
                print("Ошибка сохранения результатов! Введен неверный путь. ", e)

    def enhance_lyrics(self):
        if self.metadata:
            for song, song_info in self.metadata.items():
                if bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df]", song_info.get("lyrics"))):
                    enhanced_lyrics = research_with_musixmatch(song)
                    if enhanced_lyrics:
                        song_info['lyrics'] = enhanced_lyrics
                        print(f"Данные были успешно обновлены для песни {song}")
                    else:
                        song_info['lyrics'] = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df]", '', song_info['lyrics'])
                elif len(song_info.get("lyrics", '').split(':')) >= 2:
                    enhanced_lyrics = research_with_musixmatch(song)
                    if enhanced_lyrics:
                        song_info['lyrics'] = enhanced_lyrics
                        print(f"Данные были успешно обновлены для песни {song}")