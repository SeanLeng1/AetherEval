
import csv
import io
import json
import random
from pathlib import Path
from urllib.request import urlopen


GPQA_DIAMOND_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(GPQA_DIAMOND_CSV_URL, timeout=60) as response:  # noqa: S310
        csv_text = response.read().decode("utf-8")

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

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
