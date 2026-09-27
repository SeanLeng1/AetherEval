"""Pinned data/prompt fingerprints and gold round-trips for every native benchmark.

A fingerprint mismatch means a task's rows, ids, rendered prompts, golds or scoring
metadata (Sample.meta and Sample.data) changed. This guards data rebuilds and
refactors of load_samples, build_prompt and the shared prompt helpers; it does not
cover metrics or extraction code. After a deliberate data or prompt change, rewrite
the pins and review the diff of tests/golden/benchmark_fingerprints.json:

    AETHEREVAL_UPDATE_GOLDEN=1 python -m unittest tests.test_benchmark_fingerprints

Set AETHEREVAL_SLOW_TESTS=1 to also run every HumanEval+/MBPP+ canonical solution.
"""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

from aethereval.core.runner import _score_generation_outputs
from aethereval.core.task_register import (
    _load_module_from_path,
    discover_tasks,
    list_tasks,
    load_task,
)
from aethereval.core.types import GenerationOutput
from tests._deps import requires

GOLDEN = Path(__file__).resolve().parent / "golden" / "benchmark_fingerprints.json"
UPDATE = os.environ.get("AETHEREVAL_UPDATE_GOLDEN") == "1"

_BOXED = "The final answer is $\\boxed{{{}}}$"
# One canonical answer format per rule-graded task; its own scorer must accept the gold.
GOLD_RESPONSES = {
    **dict.fromkeys(
        ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad-bench"), _BOXED
    ),
    **dict.fromkeys(("agieval-en", "gpqa-diamond", "mmlu-pro"), "Answer: {}"),
    "bbh": "So the answer is {}.",
}
# Upstream defects: golds or canonical solutions that the task's own grader rejects.
KNOWN_GOLD_FAILURES = {
    # Upstream split these options on commas and the target is option text, not a
    # letter, so no response can match them.
    "bbh": {"movie_recommendation_00163", "ruin_names_00099", "ruin_names_00144"},
    # Since Python 3.12's compensated float sum, the canonical Newton solver misses 7 of
    # 788 plus inputs under the pinned find_zero oracle (see the task README).
    "humaneval-plus": {"HumanEval/32"} if sys.version_info >= (3, 12) else set(),
    # EvalPlus sanitize drops the canonical solution's generator helper `adjac`.
    "mbpp-plus": {"Mbpp/630"},
}


def _task_samples(task: str):
    # Only the task module: fingerprints must not need scorer dependencies (EvalPlus).
    spec = discover_tasks()[task]
    module = _load_module_from_path(f"aethereval_fingerprint_{task}", spec.task_module_path)
    return module, module.load_samples(spec.task_dir)


def _json_line(value, sort_keys: bool = False) -> bytes:
    return json.dumps(value, sort_keys=sort_keys, ensure_ascii=False).encode() + b"\n"


def _canonical_response(task: str, sample) -> str:
    code = sample.data["canonical_solution"]
    if task == "humaneval-plus":  # The HumanEval canonical solution is only the body.
        code = sample.data["prompt"] + code
    return f"```python\n{code}\n```"


class BenchmarkFingerprintTests(unittest.TestCase):
    def _fingerprint(self, task: str) -> dict:
        module, samples = _task_samples(task)
        ids = [sample.id for sample in samples]
        self.assertEqual(len(ids), len(set(ids)), "duplicate sample ids")
        prompts, payloads = hashlib.sha256(), hashlib.sha256()
        for sample in samples:
            prompt = module.build_prompt(sample)
            self.assertTrue(prompt, f"{sample.id}: empty prompt")
            # Prompts exactly as rendered (records store them); payload key order is
            # only data-file formatting.
            prompts.update(_json_line([sample.id, prompt]))
            payloads.update(
                _json_line([sample.id, sample.gold, sample.meta, sample.data], sort_keys=True)
            )
        return {
            "count": len(samples),
            "prompt_sha256": prompts.hexdigest(),
            "sample_sha256": payloads.hexdigest(),
        }

    def test_samples_and_prompts_match_pinned_fingerprints(self) -> None:
        expected = {} if UPDATE else json.loads(GOLDEN.read_text(encoding="utf-8"))
        actual = {}
        for task in list_tasks():
            with self.subTest(task=task):
                actual[task] = self._fingerprint(task)
                if not UPDATE:
                    self.assertEqual(actual[task], expected.get(task))
        if UPDATE:
            GOLDEN.parent.mkdir(exist_ok=True)
            GOLDEN.write_text(
                json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            self.assertEqual(sorted(expected), list_tasks(), "stale or missing pins")

    def test_gold_answers_pass_their_own_scorers(self) -> None:
        for task, template in GOLD_RESPONSES.items():
            with self.subTest(task=task):
                bundle = load_task(task)
                samples = bundle.task_module.load_samples(bundle.spec.task_dir)
                failed = {
                    sample.id
                    for sample in samples
                    if not bundle.metrics_module.score_generation(
                        sample, template.format(sample.gold)
                    )["is_pass"]
                }
                self.assertEqual(failed, KNOWN_GOLD_FAILURES.get(task, set()))

    @unittest.skipUnless(
        os.environ.get("AETHEREVAL_SLOW_TESTS") == "1", "set AETHEREVAL_SLOW_TESTS=1"
    )
    @requires("evalplus")
    def test_code_canonical_solutions_pass(self) -> None:
        for task in ("humaneval-plus", "mbpp-plus"):
            with self.subTest(task=task):
                bundle = load_task(task)
                samples = bundle.task_module.load_samples(bundle.spec.task_dir)
                # The runner's scoring path, so the checker gets its single-thread BLAS
                # and the same 4 GiB candidate budget as a real run.
                scored = _score_generation_outputs(
                    metrics_module=bundle.metrics_module,
                    samples_by_id={sample.id: sample for sample in samples},
                    outputs=[
                        GenerationOutput(sample.id, "", [_canonical_response(task, sample)])
                        for sample in samples
                    ],
                    metric_options={"num_proc": min(16, os.cpu_count() or 1)},
                    total_records=len(samples),
                    progress_desc=f"[{task}] canonical solutions",
                )
                failed = {
                    sample_id for sample_id, [(_, passed, _, _)] in scored.items() if not passed
                }
                self.assertEqual(failed, KNOWN_GOLD_FAILURES[task])


if __name__ == "__main__":
    unittest.main()
