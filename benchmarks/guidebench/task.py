from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample
from benchmarks.guidebench.prompts import (
    ANSWER_TEMPLATE,
    MATH_QA_TEMPLATE,
    QA_TEMPLATE,
    RE_QA_TEMPLATE,
)

TASK_NAME = "guidebench"
DATA_FILE = "data/eval.jsonl"
REVISION = "78c5bfa42facee34db31e4ba03ad2c3b5a04bbbc"
CATEGORIES = ("price", "relevance", "math", "chat", "summary", "hallu")
MC_CATEGORIES = ("chat", "summary", "hallu")


def load_samples(task_dir: Path):
    samples, seen = [], set()
    for row in read_jsonl(task_dir / DATA_FILE):
        category, sample_id = row["category"], str(row["id"])
        if category not in CATEGORIES or sample_id in seen:
            raise ValueError(f"Invalid/duplicate GuideBench sample {sample_id}")
        seen.add(sample_id)
        for key in ("Instruction", "Guidelines", "Context", "Groundtruth"):
            if key not in row:
                raise ValueError(f"GuideBench {sample_id}: missing {key}")
        field = "OptimalOption" if category in MC_CATEGORIES else "ReferenceAnswer"
        gold = row["Groundtruth"][field]
        if gold is None or gold == "":
            raise ValueError(f"GuideBench {sample_id}: empty gold")
        if category in MC_CATEGORIES and not row.get("MultipleOptions"):
            raise ValueError(f"GuideBench {sample_id}: missing choices")
        samples.append(
            Sample(
                id=sample_id,
                gold=gold,
                data={
                    key: row[key]
                    for key in (
                        "Instruction",
                        "Guidelines",
                        "Context",
                        "MultipleOptions",
                    )
                    if key in row
                },
                meta={
                    "category": category,
                    "source_revision": REVISION,
                    "source": "Dlxxx/GuideBench",
                },
            )
        )
    return samples


def build_prompt(sample: Sample):
    category = sample.meta["category"]
    template = {
        "math": MATH_QA_TEMPLATE,
        "price": QA_TEMPLATE,
        "relevance": RE_QA_TEMPLATE,
    }.get(category, ANSWER_TEMPLATE)
    # Upstream formats the Guidelines list directly, not as JSON.
    return template.format(**sample.data)
