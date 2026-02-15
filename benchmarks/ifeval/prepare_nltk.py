from __future__ import annotations

import os
from pathlib import Path


REQUIRED_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}


def main() -> None:
    try:
        import nltk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "nltk is required. Install with `pip install nltk`."
        ) from exc

    task_dir = Path(__file__).resolve().parent
    nltk_data_dir = task_dir / "ifeval_lib" / ".nltk_data"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NLTK_DATA", str(nltk_data_dir))
    nltk.data.path.insert(0, str(nltk_data_dir))
    # Force local check/download so this script really prepares offline assets.
    nltk.data.path = [str(nltk_data_dir)]

    downloaded: list[str] = []
    for resource, lookup_path in REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(resource, download_dir=str(nltk_data_dir), quiet=True)
            downloaded.append(resource)

    removed_zip_files: list[str] = []
    cleanup_pairs = [
        (nltk_data_dir / "tokenizers" / "punkt.zip", nltk_data_dir / "tokenizers" / "punkt"),
        (nltk_data_dir / "tokenizers" / "punkt_tab.zip", nltk_data_dir / "tokenizers" / "punkt_tab"),
    ]
    for zip_path, extracted_dir in cleanup_pairs:
        if zip_path.exists() and extracted_dir.exists():
            zip_path.unlink()
            removed_zip_files.append(str(zip_path))

    print(f"NLTK data dir: {nltk_data_dir}")
    if downloaded:
        print(f"downloaded: {', '.join(downloaded)}")
    else:
        print("all required resources already present")
    if removed_zip_files:
        print(f"removed zip files: {', '.join(removed_zip_files)}")


if __name__ == "__main__":
    main()
