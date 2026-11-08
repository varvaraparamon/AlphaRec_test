"""Подготовка подвыборки MSD в формате официального кода AlphaRec.

Шаги:

1. берутся ``--max-songs`` песен с наибольшим числом слушателей;
2. остаются пользователи, у которых в этом каталоге не меньше ``--min-user-songs`` песен,
   из них случайно выбираются ``--max-users``;
3. песни каждого пользователя делятся на train/valid/test (80/10/10, минимум по одной в valid и test);
4. текст «исполнитель — название» каждой песни кодируется языковой моделью.

Результат — структура, которую ожидает AlphaRec (``--data_path ./data/General/ --dataset msd``)::

    data/General/msd/
        cf_data/train.txt, valid.txt, test.txt   строки «user item item ...»
        item_info/item_cf_embeds_bge_large_array.npy
        songs.csv, users.csv                     исходные id для индексов

Пример::

    python prepare_data.py --max-users 3000 --max-songs 2000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_manager import DataManager

OUT_DIR = Path(__file__).resolve().parent / "data" / "General" / "msd"
EMBEDS_FILE = "item_cf_embeds_bge_large_array.npy"


def select_subset(
    triplets: pd.DataFrame, max_users: int, max_songs: int, min_user_songs: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Возвращает пары (user_id, song_id) после фильтрации по популярности и активности."""
    top_songs = triplets["song_id"].value_counts().head(max_songs).index
    pairs = triplets.loc[triplets["song_id"].isin(top_songs), ["user_id", "song_id"]]
    pairs = pairs.astype(str)
    counts = pairs["user_id"].value_counts()
    eligible = counts[counts >= min_user_songs].index.to_numpy()
    if len(eligible) == 0:
        raise ValueError("После фильтрации не осталось пользователей")
    if len(eligible) > max_users:
        eligible = rng.choice(eligible, size=max_users, replace=False)
    return pairs[pairs["user_id"].isin(eligible)].reset_index(drop=True)


def split_per_user(users: np.ndarray, rng: np.random.Generator, fraction: float = 0.1) -> np.ndarray:
    """Для каждого взаимодействия возвращает его часть: ``train``, ``valid`` или ``test``."""
    part = np.empty(len(users), dtype="<U5")
    for user in np.unique(users):
        idx = rng.permutation(np.flatnonzero(users == user))
        n_held = max(1, int(len(idx) * fraction))
        part[idx[:n_held]] = "test"
        part[idx[n_held : 2 * n_held]] = "valid"
        part[idx[2 * n_held :]] = "train"
    return part


def write_user_lists(path: Path, users: np.ndarray, items: np.ndarray) -> None:
    """Пишет файл в формате AlphaRec: по строке на пользователя, «user item item ...»."""
    frame = pd.DataFrame({"user": users, "item": items}).sort_values(["user", "item"])
    with open(path, "w") as fh:
        for user, group in frame.groupby("user"):
            fh.write(" ".join(map(str, [user, *group["item"]])) + "\n")


def encode_items(texts: list[str], model_name: str) -> np.ndarray:
    """Кодирует описания песен моделью sentence-transformers (L2-нормированные векторы)."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def main() -> None:
    """Точка входа: строит подвыборку, разбиение и эмбеддинги и сохраняет их в ``data/``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-users", type=int, default=3000)
    parser.add_argument("--max-songs", type=int, default=2000)
    parser.add_argument("--min-user-songs", type=int, default=5)
    parser.add_argument("--lm-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dm = DataManager()
    pairs = select_subset(dm.triplets(), args.max_users, args.max_songs, args.min_user_songs, rng)

    user_ids, users = np.unique(pairs["user_id"].to_numpy(), return_inverse=True)
    song_ids, items = np.unique(pairs["song_id"].to_numpy(), return_inverse=True)
    part = split_per_user(users, rng)

    (OUT_DIR / "cf_data").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "item_info").mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid", "test"):
        mask = part == name
        write_user_lists(OUT_DIR / "cf_data" / f"{name}.txt", users[mask], items[mask])
        print(f"{name}: {mask.sum()} взаимодействий")

    tracks = dm.tracks().drop_duplicates("song_id").set_index("song_id")
    songs = tracks.reindex(song_ids)[["artist", "title"]].fillna("")
    songs.index.name = "song_id"
    songs.to_csv(OUT_DIR / "songs.csv")
    pd.Series(user_ids, name="user_id").to_csv(OUT_DIR / "users.csv", index=False)

    texts = (songs["artist"] + " - " + songs["title"]).tolist()
    embeds = encode_items(texts, args.lm_model)
    np.save(OUT_DIR / "item_info" / EMBEDS_FILE, embeds.astype(np.float32))
    print(f"{len(user_ids)} пользователей, {len(song_ids)} песен, эмбеддинги {embeds.shape} ({args.lm_model})")


if __name__ == "__main__":
    main()
