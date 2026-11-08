"""Загрузка исходных таблиц Million Song Dataset с кэшированием в pickle."""

from pathlib import Path

import numpy as np
import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DATA_CACHE_DIR = DATASETS_DIR / "cache"
TRACKS_FILE = DATASETS_DIR / "p02_unique_tracks.txt"
TRIPLETS_FILE = DATASETS_DIR / "train_triplets.txt"


class DataManager:
    """Ленивая загрузка таблиц треков и прослушиваний.

    При первом обращении таблица читается из текстового файла и сохраняется
    в pickle-кэш, последующие запуски читают уже кэш.
    """

    def __init__(
        self,
        cache_dir: Path = DATA_CACHE_DIR,
        tracks_file: Path = TRACKS_FILE,
        triplets_file: Path = TRIPLETS_FILE,
    ):
        """Создаёт менеджер данных.

        Args:
            cache_dir: папка для pickle-кэша.
            tracks_file: файл ``unique_tracks`` (track_id, song_id, artist, title).
            triplets_file: файл ``train_triplets`` (user_id, song_id, play_count).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_file = Path(tracks_file)
        self.triplets_file = Path(triplets_file)
        self._tracks: pd.DataFrame | None = None
        self._triplets: pd.DataFrame | None = None

    @staticmethod
    def _require(path: Path) -> Path:
        """Проверяет, что файл датасета существует."""
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл датасета: {path}")
        return path

    def tracks(self) -> pd.DataFrame:
        """Возвращает таблицу треков: track_id, song_id, artist, title."""
        if self._tracks is not None:
            return self._tracks
        cache = self.cache_dir / "tracks.pkl"
        if cache.exists():
            self._tracks = pd.read_pickle(cache)
            return self._tracks
        df = pd.read_csv(
            self._require(self.tracks_file),
            sep="<SEP>",
            engine="python",
            header=None,
            names=["track_id", "song_id", "artist", "title"],
        )
        df.to_pickle(cache)
        self._tracks = df
        return df

    def triplets(self) -> pd.DataFrame:
        """Возвращает таблицу прослушиваний: user_id, song_id, play_count.

        Идентификаторы хранятся как ``category`` для экономии памяти.
        """
        if self._triplets is not None:
            return self._triplets
        cache = self.cache_dir / "triplets.pkl"
        if cache.exists():
            self._triplets = pd.read_pickle(cache)
            return self._triplets
        df = pd.read_csv(
            self._require(self.triplets_file),
            sep="\t",
            header=None,
            names=["user_id", "song_id", "play_count"],
            dtype={"user_id": "category", "song_id": "category"},
        )
        df["play_count"] = df["play_count"].astype(np.int32)
        df.to_pickle(cache)
        self._triplets = df
        return df
