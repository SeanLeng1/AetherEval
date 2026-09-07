"""Paired weight requests; scoring sees the unconditioned conversation."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample

TASK_NAME = "safe-alignment-dynamic"
DATA_FILE = "data/eval.jsonl"


def protocol_hash(protocol):
    payload = json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_protocol(task_dir: Path):
    return json.loads((task_dir / "data/protocol.json").read_text())


def targets_for_weight(weights, artifact):
    mapping = artifact["target_mapping"]
    keys = [c["key"] for c in artifact["components"]]
    if mapping["method"] != "ric_p2_quantile" or mapping["components"] != keys:
        raise ValueError("Target mapping must match the training component order")
    w = np.asarray(weights, dtype=float)
    if (
        w.shape != (len(keys),)
        or not np.isfinite(w).all()
        or (w < 0).any()
        or not np.isclose(w.sum(), 1)
    ):
        raise ValueError("Expected finite simplex weights")
    low, high = np.asarray(mapping["low"]), np.asarray(mapping["high"])
    return (low + (high - low) * w / np.linalg.norm(w)).tolist()


def load_samples(task_dir: Path) -> list[Sample]:
    protocol = load_protocol(task_dir)
    digest = protocol_hash(protocol)
    artifact = protocol["score_conditioning"]
    samples = []
    seen = set()
    for row in read_jsonl(task_dir / DATA_FILE):
        key = (row["subset"], row["prompt_id"])
        if key in seen:
            raise ValueError(f"Duplicate evaluation problem: {key}")
        seen.add(key)
        for condition, weights in enumerate(protocol["weights"] + [None]):
            targets = None if weights is None else targets_for_weight(weights, artifact)
            samples.append(
                Sample(
                    id=f"{row['subset']}:{row['prompt_id']}@{condition}",
                    data={"prompt": row["messages"], "artifact": artifact},
                    meta={
                        "data_source": row["subset"],
                        "problem_id": row["prompt_id"],
                        "condition": condition,
                        "weights": weights,
                        "targets": targets,
                        "protocol_hash": digest,
                    },
                )
            )
    return samples


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    messages = copy.deepcopy(sample.data["prompt"])
    targets = sample.meta["targets"]
    if targets is not None:
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("Score conditioning requires a final user message")
        fields = [
            f"{component['label']}={round(float(value), 1) + 0.0:.1f}"
            for component, value in zip(
                sample.data["artifact"]["components"], targets, strict=True
            )
        ]
        messages[-1]["content"] += "\n\nTarget scores: " + ", ".join(fields)
    return messages
