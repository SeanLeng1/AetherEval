from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "researchqa"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / DATA_FILE):
        sample_id = str(row["id"])
        rubrics = row["rubric"]
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError(f"ResearchQA sample {sample_id} has no rubric")
        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                data={
                    "query": str(row["query"]),
                    "date": str(row["date"]),
                    "rubric": rubrics,
                },
                meta={
                    "general_domain": str(row["general_domain"]),
                    "subdomain": str(row["subdomain"]),
                    "field": str(row["field"]),
                    "date": str(row["date"]),
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    return (
        f"{sample.data['query']}\n\n"
        "Provide a citation-supported answer of approximately 250 words. "
        f"Do not use or cite sources published after {sample.data['date']}."
    )
