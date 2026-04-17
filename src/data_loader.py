from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import pandas as pd

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
DATASET_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}
DATASET_MD5 = {
    "train": "b1a00f826fdfbd249f79597b59e1dc12",
    "test": "d52ea96a97a2d943681189a97654912d",
}
RAW_COLUMNS = ["label_id", "title", "description"]


def clean_text(text: str) -> str:
    normalized = str(text).replace("\\n", " ").replace("\n", " ")
    return " ".join(normalized.split())


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    with urlopen(url) as response, temporary_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temporary_path.replace(destination)


def ensure_dataset_downloaded(data_dir: Path, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str | int]] = {}

    for split_name, url in DATASET_URLS.items():
        split_path = data_dir / f"{split_name}.csv"
        expected_md5 = DATASET_MD5[split_name]

        needs_download = force or not split_path.exists()
        if split_path.exists() and file_md5(split_path) != expected_md5:
            needs_download = True

        if needs_download:
            download_file(url=url, destination=split_path)

        actual_md5 = file_md5(split_path)
        if actual_md5 != expected_md5:
            raise ValueError(f"Checksum mismatch for {split_name}.csv: expected {expected_md5}, found {actual_md5}.")

        manifest[split_name] = {
            "url": url,
            "path": str(split_path),
            "md5": actual_md5,
            "size_bytes": split_path.stat().st_size,
        }

    manifest_path = data_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_ag_news_split(path: Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=RAW_COLUMNS, keep_default_na=False)
    frame["label_id"] = frame["label_id"].astype(int) - 1
    frame["label_name"] = frame["label_id"].map(dict(enumerate(LABEL_NAMES)))
    frame["title"] = frame["title"].map(clean_text)
    frame["description"] = frame["description"].map(clean_text)
    frame["text"] = [
        ". ".join(part for part in (title, description) if part)
        for title, description in zip(frame["title"], frame["description"])
    ]
    frame["split"] = split_name
    return frame


def load_ag_news_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = load_ag_news_split(data_dir / "train.csv", "train")
    test_frame = load_ag_news_split(data_dir / "test.csv", "test")
    return train_frame, test_frame


def select_labeled_subset(train_frame: pd.DataFrame, labeled_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must be between 0 and 1.")

    labeled_subset = (
        train_frame.groupby("label_id", group_keys=False)
        .sample(frac=labeled_fraction, random_state=seed)
        .sort_index()
    )
    unlabeled_pool = train_frame.drop(index=labeled_subset.index).sort_index()
    return labeled_subset, unlabeled_pool


def build_label_map(label_names: Iterable[str] = LABEL_NAMES) -> dict[int, str]:
    return {index: label_name for index, label_name in enumerate(label_names)}
