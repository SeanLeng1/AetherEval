from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "bbh"
DATA_FILE = "data/eval.jsonl"


_REQUIRED_KEYS = {
    "id",
    "subset",
    "input",
    "target",
    "answer",
    "description",
}


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)
    if not rows:
        raise RuntimeError(
            "BBH data file is empty or missing. Run "
            "`python benchmarks/bbh/prepare_data.py` to generate `data/eval.jsonl`."
        )

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("BBH row must be a JSON object")

        missing = sorted(_REQUIRED_KEYS - set(row.keys()))
        if missing:
            raise ValueError(f"BBH row missing keys: {', '.join(missing)}")

        sample_id = str(row["id"]).strip()
        subset = str(row["subset"]).strip()
        input_text = str(row["input"]).strip()
        target = str(row["target"]).strip()
        answer = str(row["answer"]).strip()

        if not sample_id:
            raise ValueError("BBH sample id must be non-empty")
        if not subset:
            raise ValueError(f"BBH subset is empty for sample {sample_id}")
        if not input_text:
            raise ValueError(f"BBH input is empty for sample {sample_id}")
        if not target:
            raise ValueError(f"BBH target is empty for sample {sample_id}")
        if not answer:
            raise ValueError(f"BBH answer is empty for sample {sample_id}")

        description = str(row["description"]).strip()

        samples.append(
            Sample(
                id=sample_id,
                gold=answer,
                meta={
                    "subset": subset,
                    "source": str(row.get("source", "lukaemon/bbh")).strip(),
                },
                data={
                    "subset": subset,
                    "input": input_text,
                    "target": target,
                    "answer": answer,
                    "description": description,
                },
            )
        )

    return samples


def build_prompt(sample: Sample) -> str:
    description = str(sample.data.get("description", "")).strip()
    input_text = str(sample.data["input"]).strip()

    parts: list[str] = []
    if description:
        parts.append(description)
    parts.append(f"Question: {input_text}")
    parts.append("Answer: Let's think step by step.")
    return "\n\n".join(parts)
