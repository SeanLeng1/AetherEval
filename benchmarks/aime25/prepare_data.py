from pathlib import Path

from benchmark_utils.aime import prepare_aime_dataset


def main() -> None:
    prepare_aime_dataset("yentinglin/aime_2025", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
