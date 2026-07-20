from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "writingbench"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / DATA_FILE):
        sample_id = str(row["index"])
        checklist = row["checklist"]
        if not isinstance(checklist, list) or not checklist:
            raise ValueError(f"WritingBench sample {sample_id} has no checklist")
        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                data={"query": str(row["query"]), "checklist": checklist},
                meta={
                    "domain1": str(row["domain1"]),
                    "domain2": str(row["domain2"]),
                    "lang": str(row.get("lang", "")),
                    "requirement_subsets": [
                        str(item) for item in row.get("requirement_subsets", [])
                    ],
                    "requirement_criteria": {
                        str(name): [str(item) for item in criteria]
                        for name, criteria in row.get(
                            "requirement_criteria", {}
                        ).items()
                    },
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    return str(sample.data["query"])
