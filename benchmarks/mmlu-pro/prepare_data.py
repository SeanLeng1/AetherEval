from pathlib import Path
from string import ascii_uppercase

from benchmark_utils.data import load_hf, write_task_jsonl


REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"


def main() -> None:
    ds = load_hf("TIGER-Lab/MMLU-Pro", "default", "test", REVISION)

    rows: list[dict[str, object]] = []
    for idx, row in enumerate(ds):
        question = str(row["question"]).strip()
        raw_options = [str(opt).strip() for opt in row["options"]]
        answer_index = int(row["answer_index"])

        filtered_options: list[str] = []
        filtered_answer_index: int | None = None
        for opt_idx, opt in enumerate(raw_options):
            if not opt or opt.upper() == "N/A":
                continue
            if opt_idx == answer_index:
                filtered_answer_index = len(filtered_options)
            filtered_options.append(opt)

        if not question or len(filtered_options) < 2:
            raise ValueError(f"Invalid source row at index {idx}")

        if filtered_answer_index is None:
            answer_letter = str(row.get("answer", "")).strip().upper()
            if answer_letter not in ascii_uppercase:
                raise ValueError(f"Cannot resolve answer for row {idx}")
            filtered_answer_index = ascii_uppercase.index(answer_letter)

        if filtered_answer_index < 0 or filtered_answer_index >= len(filtered_options):
            raise ValueError(f"Answer index out of range at row {idx}")

        choices = {
            ascii_uppercase[i]: filtered_options[i]
            for i in range(len(filtered_options))
        }
        answer = ascii_uppercase[filtered_answer_index]

        question_id = int(row.get("question_id", idx))
        rows.append(
            {
                "id": str(question_id),
                "question_id": question_id,
                "question": question,
                "choices": choices,
                "answer": answer,
                "category": str(row.get("category", "")).strip(),
                "src": str(row.get("src", "")).strip(),
                "source": "TIGER-Lab/MMLU-Pro",
            }
        )

    write_task_jsonl(Path(__file__).resolve().parent, rows)


if __name__ == "__main__":
    main()
