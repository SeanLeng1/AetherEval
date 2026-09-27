import csv
import io
import random
from pathlib import Path

from benchmark_utils.data import read_text, write_task_jsonl


GPQA_DIAMOND_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
)
GPQA_DIAMOND_CSV_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def main() -> None:
    csv_text = read_text(GPQA_DIAMOND_CSV_URL, sha256=GPQA_DIAMOND_CSV_SHA256)

    reader = csv.DictReader(io.StringIO(csv_text))
    rng = random.Random(0)

    rows: list[dict[str, object]] = []
    for idx, row in enumerate(reader):
        question = _clean(row.get("Question"))
        correct = _clean(row.get("Correct Answer"))
        incorrect = [
            _clean(row.get("Incorrect Answer 1")),
            _clean(row.get("Incorrect Answer 2")),
            _clean(row.get("Incorrect Answer 3")),
        ]

        if not question or not correct or any(not x for x in incorrect):
            raise ValueError(f"Invalid source row at index {idx}")

        all_choices = [correct] + incorrect
        permutation = [0, 1, 2, 3]
        rng.shuffle(permutation)
        shuffled = [all_choices[i] for i in permutation]
        answer = "ABCD"[permutation.index(0)]

        record_id = _clean(row.get("Record ID")) or f"row_{idx:04d}"
        rows.append(
            {
                "id": record_id,
                "record_id": record_id,
                "question": question,
                "choices": {
                    "A": shuffled[0],
                    "B": shuffled[1],
                    "C": shuffled[2],
                    "D": shuffled[3],
                },
                "answer": answer,
                "correct_answer": correct,
                "domain": _clean(row.get("High-level domain")),
                "subdomain": _clean(row.get("Subdomain")),
                "source": "openaipublic/simple-evals/gpqa_diamond.csv",
            }
        )

    write_task_jsonl(Path(__file__).resolve().parent, rows)


if __name__ == "__main__":
    main()
