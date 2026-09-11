"""Freeze held-out prompts, a common weight grid, and the TRAIN score mapping."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aethereval.core.io import write_json, write_jsonl

DATASET = "RLLab/safe-alignment-dynamic"
COMPONENTS = [
    {"key": "helpful", "field": "reward_helpful", "label": "helpfulness"},
    {"key": "harmless", "field": "reward_harmless", "label": "harmlessness"},
]
SUBSETS = ["alpaca", "hh-rlhf", "pku-saferlhf"]
# Match Dynamic RL's unfiltered validation source and stable row identities.
TEST_ROOT = (
    "https://raw.githubusercontent.com/Qwen-Applications/GD2PO/"
    "f1ad765bc9a330e6cf387f95e9c1e5a6c4bb2d02/safe-alignment/dataset"
)
TEST_FILES = {
    "alpaca": "alpaca_prompt_only/test.parquet",
    "hh-rlhf": "anthropic_hh_rlhf/test.parquet",
    "pku-saferlhf": "pku-saferlhf/test.parquet",
}


def load_artifact(rl_data=None, revision=None):
    if rl_data is not None:
        import pyarrow.parquet as pq

        row = next(
            pq.ParquetFile(rl_data / "train.parquet").iter_batches(
                batch_size=1, columns=["extra_info"]
            )
        ).to_pylist()[0]
        return json.loads(row["extra_info"]["score_conditioning"])

    from huggingface_hub import HfApi, hf_hub_download

    revision = revision or HfApi().dataset_info(DATASET).sha
    path = hf_hub_download(
        DATASET, "sft-prompts/scoring/train.json", repo_type="dataset", revision=revision
    )
    score_stats = json.loads(Path(path).read_text())
    if score_stats["config"] != "sft-prompts" or score_stats["split"] != "train":
        raise ValueError("Expected scoring statistics from sft-prompts/train")
    models = [score_stats["models"][c["key"]] for c in COMPONENTS]
    means = np.asarray([model["mean"] for model in models], dtype=float)
    stds = np.asarray([model["std"] for model in models], dtype=float)
    quantiles = np.asarray([[model[q] for model in models] for q in ("q05", "q95")], dtype=float)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("Invalid training score statistics")
    # Positive affine normalization commutes with the stored response-level quantiles.
    low, high = (quantiles - means) / stds
    if not np.isfinite([low, high]).all():
        raise ValueError("Invalid training score quantiles")
    if np.any(high <= low):
        raise ValueError("Constant score range cannot support target mapping")
    return {
        "task": "safe-alignment",
        "dataset": DATASET,
        "revision": revision,
        "components": COMPONENTS,
        "score_stats": score_stats,
        "target_mapping": {
            "method": "ric_p2_quantile",
            "components": [c["key"] for c in COMPONENTS],
            "quantiles": [0.05, 0.95],
            "low": low.tolist(),
            "high": high.tolist(),
        },
    }


def load_test(subset):
    from datasets import load_dataset

    rows = load_dataset(
        "parquet", data_files={"test": f"{TEST_ROOT}/{TEST_FILES[subset]}"}, split="test"
    )
    return [
        {
            "prompt_id": f"{subset}/test/{index}",
            "subset": subset,
            "split": "test",
            "messages": row["prompt"],
        }
        for index, row in enumerate(rows)
    ]


def select_problems(rows, *, limit, seed):
    unique = {}
    for row in rows:
        if row["split"] != "test":
            raise ValueError("Only held-out test rows may enter evaluation")
        key = str(row["prompt_id"])
        problem = {
            "prompt_id": key,
            "subset": row["subset"],
            "messages": row["messages"],
        }
        if key in unique and unique[key] != problem:
            raise ValueError(f"Conflicting prompt identity: {key}")
        unique[key] = problem
    ordered = sorted(
        unique, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest()
    )
    return [unique[key] for key in (ordered[:limit] if limit else ordered)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rl-data",
        type=Path,
        help="Reuse exact score statistics and target mapping from RL train.parquet",
    )
    parser.add_argument(
        "--revision",
        help="HF revision used for SFT; default resolves current revision once",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=0,
        help="Unique held-out problems per source; default 0 means all",
    )
    parser.add_argument("--num-weights", type=int, default=5)
    parser.add_argument("--weight-mode", choices=["grid", "dirichlet"], default="grid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_weights < 2 or args.limit_per_source < 0:
        parser.error("Need at least two weights and a nonnegative problem limit")
    if args.rl_data is not None and args.revision is not None:
        parser.error("--rl-data already pins the training dataset revision")
    artifact = load_artifact(args.rl_data, args.revision)
    if artifact["task"] != "safe-alignment" or artifact["components"] != COMPONENTS:
        raise ValueError("Expected the safe-alignment training score contract")
    if artifact["score_stats"]["harmless_sign"] != 1:
        raise ValueError("Harmless reward scoring requires the +1 training convention")
    weights = (
        np.column_stack(
            (np.linspace(0, 1, args.num_weights), np.linspace(1, 0, args.num_weights))
        )
        if args.weight_mode == "grid"
        else np.random.default_rng(args.seed).dirichlet([1, 1], args.num_weights)
    )
    weights = weights[np.argsort(weights[:, 0])].tolist()
    problems = []
    for subset in SUBSETS:
        selected = select_problems(load_test(subset), limit=args.limit_per_source, seed=args.seed)
        if not selected:
            raise ValueError(f"Empty evaluation source: {subset}")
        problems.extend(selected)
        print(
            f"{subset}: {len(selected)} problems x {len(weights) + 1} conditions (including no target)"
        )
    output = Path(__file__).parent / "data"
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "eval.jsonl", problems)
    write_json(
        output / "protocol.json",
        {
            "score_conditioning": artifact,
            "weights": weights,
            "weight_mode": args.weight_mode,
            "seed": args.seed,
            "limit_per_source": args.limit_per_source,
        },
    )


if __name__ == "__main__":
    main()
