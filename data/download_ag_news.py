from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ensure_dataset_downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the AG News train and test CSV files.")
    parser.add_argument("--force", action="store_true", help="Redownload the raw CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dataset_downloaded(PROJECT_ROOT / "data", force=args.force)


if __name__ == "__main__":
    main()
