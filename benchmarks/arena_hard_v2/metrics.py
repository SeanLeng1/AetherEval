import json
import re
from pathlib import Path
from typing import Any

import tiktoken

from aethereval.core.types import GenerationOutput, Sample
from aethereval.core.task_defaults import resolve_task_default_metrics
from benchmark_utils.llm_judge import (
    NORMAL_FORMAT_ATTEMPTS,
    chat_completion,
    local_constraint_body,
    parallel_map,
    parse_json_object,
    resolve_judge_settings,
)


PRIMARY_METRIC = "style_controlled_win_rate"
USES_LLM_JUDGE = True
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = str(
    resolve_task_default_metrics("arena_hard_v2").get("judge_model", "gpt-4.1")
)
BASELINE_MODEL = "o3-mini-2025-01-31"
ROOT = Path(__file__).resolve().parent
STYLE_COHORT_FILE = ROOT / "data/style_cohort.jsonl"
PROMPT_TEMPLATE = "<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"
SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""
LABEL_TO_SCORE = {
    "A>B": [1.0],
    "A>>B": [1.0] * 3,
    "A=B": [0.5],
    "A<<B": [0.0] * 3,
    "A<B": [0.0],
    "B>A": [0.0],
    "B>>A": [0.0] * 3,
    "B=A": [0.5],
    "B<<A": [1.0] * 3,
    "B<A": [1.0],
}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["A>>B", "A>B", "A=B", "B>A", "B>>A"]},
    },
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}


def validate_metric_options(metric_options: dict[str, Any] | None = None) -> None:
    resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)
    if not STYLE_COHORT_FILE.exists():
        raise FileNotFoundError(
            f"Arena-Hard style cohort missing: {STYLE_COHORT_FILE}. Run prepare_data.py."
        )


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    del sample, generation
    raise RuntimeError("Arena-Hard-v2 requires batched pairwise LLM-judge scoring")


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    settings = resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)
    jobs: list[tuple[int, int, int, str]] = []
    candidate_metadata: list[list[dict[str, Any]]] = []
    for sample_idx, (sample, output) in enumerate(
        zip(samples, generation_outputs, strict=True)
    ):
        if sample.id != output.sample_id:
            raise ValueError("Arena-Hard-v2 sample/output mismatch")
        per_sample_metadata: list[dict[str, Any]] = []
        for gen_idx, generation in enumerate(output.generations):
            per_sample_metadata.append(_style_metadata(generation))
            baseline = sample.data["baseline_answer"]
            jobs.append(
                (
                    sample_idx,
                    gen_idx,
                    0,
                    PROMPT_TEMPLATE.format(
                        question=sample.data["prompt"],
                        answer_a=baseline,
                        answer_b=generation,
                    ),
                )
            )
            jobs.append(
                (
                    sample_idx,
                    gen_idx,
                    1,
                    PROMPT_TEMPLATE.format(
                        question=sample.data["prompt"],
                        answer_a=generation,
                        answer_b=baseline,
                    ),
                )
            )
        candidate_metadata.append(per_sample_metadata)

    def judge(job: tuple[int, int, int, str]) -> dict[str, Any]:
        _, _, _, prompt = job
        last_text = ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for _ in range(NORMAL_FORMAT_ATTEMPTS):
            last_text = chat_completion(
                settings,
                messages,
            )
            label = _parse_label(last_text)
            if label is not None:
                return {"score": label, "judgment": last_text}

        constraint = local_constraint_body(settings, json_schema=VERDICT_SCHEMA)
        if constraint is not None:
            try:
                last_text = chat_completion(
                    settings,
                    messages,
                    extra_body=constraint,
                )
                label = _parse_label(last_text)
                if label is not None:
                    return {"score": label, "judgment": last_text}
            except (RuntimeError, ValueError):
                pass

        # Official Arena-Hard stores null and filters the whole judgment row.
        return {
            "score": None,
            "judgment": last_text,
            "error": "judge returned no parseable verdict",
        }

    games = parallel_map(
        judge, jobs, workers=settings.workers, desc="Arena-Hard-v2 judge"
    )
    results: list[list[dict[str, Any]]] = []
    offset = 0
    for sample_idx, output in enumerate(generation_outputs):
        per_sample: list[dict[str, Any]] = []
        for gen_idx in range(len(output.generations)):
            game0, game1 = games[offset : offset + 2]
            offset += 2
            judge_failed = game0["score"] is None or game1["score"] is None
            battle_scores: list[float] = []
            if not judge_failed:
                battle_scores = LABEL_TO_SCORE[game1["score"]] + [
                    1.0 - value for value in LABEL_TO_SCORE[game0["score"]]
                ]
            score = sum(battle_scores) / len(battle_scores) if battle_scores else 0.0
            per_sample.append(
                {
                    "score": score,
                    "is_pass": score > 0.5,
                    "parsed": [game0, game1],
                    "meta": {
                        "battle_scores": battle_scores,
                        "judge_failed": judge_failed,
                        "candidate_metadata": candidate_metadata[sample_idx][gen_idx],
                    },
                }
            )
        results.append(per_sample)
    return results


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = metric_options or {}
    baseline_by_uid = {
        str(sample["sample_id"]): sample["meta"]["baseline_metadata"]
        for sample in sample_results
    }
    candidate_rows: list[dict[str, Any]] = []
    candidate_scores: list[float] = []
    judge_failures = 0
    for sample in sample_results:
        uid = str(sample["sample_id"])
        for record in sample.get("records", []):
            meta = record.get("meta", {})
            if meta.get("judge_failed"):
                judge_failures += 1
                continue
            for score in meta["battle_scores"]:
                candidate_scores.append(float(score))
                candidate_rows.append(
                    {
                        "uid": uid,
                        "model": "__candidate__",
                        "score": float(score),
                        "model_metadata": meta["candidate_metadata"],
                        "baseline_metadata": baseline_by_uid[uid],
                    }
                )

    cohort_rows: list[dict[str, Any]] = []
    with STYLE_COHORT_FILE.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["uid"] not in baseline_by_uid:
                continue
            for score in row["battle_scores"]:
                cohort_rows.append(
                    {
                        "uid": row["uid"],
                        "model": row["model"],
                        "score": float(score),
                        "model_metadata": row["model_metadata"],
                        "baseline_metadata": baseline_by_uid[row["uid"]],
                    }
                )

    if candidate_rows:
        median, lower, upper = _style_controlled_score(
            cohort_rows + candidate_rows,
            target_model="__candidate__",
            seed=int(options.get("bootstrap_seed", 42)),
            rounds=100,
        )
    else:
        median = lower = upper = 0.0
    metrics: dict[str, Any] = {
        "style_controlled_win_rate": median * 100.0,
        "style_controlled_ci_lower": lower * 100.0,
        "style_controlled_ci_upper": upper * 100.0,
        "raw_win_rate": _mean(candidate_scores) * 100.0,
        "scored_judgments": float(len(candidate_scores)),
        "judge_failures": float(judge_failures),
    }
    if judge_failures:
        metrics["__warnings__"] = [
            f"{judge_failures} Arena-Hard judgments were unparseable and excluded, "
            "matching the official result loader."
        ]
    return metrics


def _parse_label(text: str) -> str | None:
    try:
        parsed = parse_json_object(text)
    except (ValueError, TypeError):
        parsed = {}
    verdict = str(parsed.get("verdict", "")).upper()
    if verdict in LABEL_TO_SCORE:
        return verdict

    upper = text.upper()
    for pattern in (r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"):
        matches = [match for match in re.findall(pattern, upper) if match]
        if matches and matches[-1] in LABEL_TO_SCORE:
            return matches[-1]
    return None


def _style_metadata(text: str) -> dict[str, Any]:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    without_code = text
    for block in re.findall(r"```([^`]*)```", text):
        without_code = without_code.replace(block, "")
    return {
        "token_len": len(encoding.encode(text, disallowed_special=())),
        "header_count": {
            f"h{level}": len(
                re.findall(rf"^#{{{level}}}\s", without_code, re.MULTILINE)
            )
            for level in range(1, 7)
        },
        "list_count": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", without_code, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", without_code, re.MULTILINE)),
        },
        "bold_count": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", without_code)),
            "__": len(re.findall(r"__[^_\n]+__", without_code)),
        },
    }


def _metadata_vector(metadata: dict[str, Any]) -> list[float]:
    return [
        float(metadata["token_len"]),
        float(sum(metadata["header_count"].values())),
        float(sum(metadata["list_count"].values())),
        float(sum(metadata["bold_count"].values())),
    ]


def _style_controlled_score(
    rows: list[dict[str, Any]],
    *,
    target_model: str,
    seed: int,
    rounds: int,
) -> tuple[float, float, float]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    import torch.optim as optim

    model_values = [row["model"] for row in rows]
    models = sorted(set(model_values + [BASELINE_MODEL]))
    model_to_idx = {model: idx for idx, model in enumerate(models)}
    one_hot = torch.zeros((len(rows), len(models)), dtype=torch.float32)
    for row_idx, model in enumerate(model_values):
        one_hot[row_idx, model_to_idx[model]] = 1.0
        one_hot[row_idx, model_to_idx[BASELINE_MODEL]] = -1.0

    model_meta = torch.tensor(
        [_metadata_vector(row["model_metadata"]) for row in rows], dtype=torch.float32
    )
    baseline_meta = torch.tensor(
        [_metadata_vector(row["baseline_metadata"]) for row in rows],
        dtype=torch.float32,
    )
    features = torch.zeros_like(model_meta)
    features[:, 0] = (model_meta[:, 0] - baseline_meta[:, 0]) / (
        model_meta[:, 0] + baseline_meta[:, 0]
    )
    model_density = model_meta[:, 1:] / (model_meta[:, :1] + 1.0)
    baseline_density = baseline_meta[:, 1:] / (baseline_meta[:, :1] + 1.0)
    features[:, 1:] = (model_density - baseline_density) / (
        model_density + baseline_density + 1.0
    )
    std = torch.std(features, axis=0)
    if (std == 0).any():
        raise ValueError("Arena-Hard style feature has zero variance")
    features = (features - torch.mean(features, axis=0)) / std
    all_features = torch.cat([one_hot, features], dim=1)
    outcomes = torch.tensor([row["score"] for row in rows], dtype=torch.float32)

    def fit(indices: Any) -> Any:
        x = all_features[indices]
        y = outcomes[indices]

        class BTModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logits = nn.Parameter(
                    nn.init.constant_(torch.empty(x.shape[1]), 0.5)
                )

        model = BTModel()
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=0.1,
            max_iter=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-9,
        )

        def closure() -> Any:
            optimizer.zero_grad()
            loss = functional.binary_cross_entropy_with_logits(
                x @ model.logits, y, reduction="sum"
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        return model.logits.detach()[:-4]

    # These are many small CPU fits. Large BLAS thread pools make each LBFGS
    # closure dramatically slower through scheduling overhead; one thread keeps
    # the exact objective/optimizer while avoiding that overhead.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        rng = np.random.default_rng(seed)
        coefs = torch.stack(
            [
                fit(torch.tensor(rng.integers(0, len(rows), size=len(rows))))
                for _ in range(rounds)
            ]
        )
    finally:
        torch.set_num_threads(previous_threads)
    target = model_to_idx[target_model]
    baseline = model_to_idx[BASELINE_MODEL]
    probabilities = torch.exp(coefs[:, target]) / (
        torch.exp(coefs[:, target]) + torch.exp(coefs[:, baseline])
    )
    return (
        float(torch.quantile(probabilities, 0.5)),
        float(torch.quantile(probabilities, 0.05)),
        float(torch.quantile(probabilities, 0.95)),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
