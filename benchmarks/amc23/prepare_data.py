from pathlib import Path

from benchmark_utils.eval_set_math import prepare_eval_set_math_dataset


def main() -> None:
    prepare_eval_set_math_dataset("amc23", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
