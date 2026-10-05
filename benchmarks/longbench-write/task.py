from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample

TASK_NAME = "longbench-write"
DATA_FILE = "data/eval.jsonl"
REVISION = "447539b356a8b09760b51eca876e19b6fc1f2dd7"


def load_samples(task_dir: Path):
    samples = []
    for index, row in enumerate(read_jsonl(task_dir / DATA_FILE)):
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"LongBench-Write {index}: missing prompt")
        if type(row.get("length")) is not int or row["length"] <= 0:
            raise ValueError(f"LongBench-Write {index}: invalid requested word count")
        if row.get("language") not in ("en", "zh"):
            raise ValueError(f"LongBench-Write {index}: invalid language label")
        samples.append(
            Sample(
                id=f"{index:03d}",
                data=row,
                meta={
                    "type": row["type"],
                    "requested_words": row["length"],
                    "language": row["language"],
                    "source": "THUDM/LongWriter",
                    "source_revision": REVISION,
                },
            )
        )
    return samples


def build_prompt(sample: Sample):
    return sample.data["prompt"]
