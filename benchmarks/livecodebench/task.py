from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

from aethereval.io import read_jsonl
from aethereval.types import Sample


TASK_NAME = "livecodebench"
DATA_FILE = "data/eval.jsonl"
DEFAULT_GEN = {
    "n": 16,
    "max_new_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.95,
}

_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write a correct Python solution that follows the required I/O format "
    "and passes all hidden tests."
)


def _ensure_str_list(value: Any, key: str, sample_id: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list for sample {sample_id}")
    out: list[str] = []
    for idx, item in enumerate(value):
        text = str(item)
        if text == "":
            raise ValueError(f"{key}[{idx}] is empty for sample {sample_id}")
        out.append(text)
    return out


def _safe_json_loads(raw: Any, default: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return default
    if raw is None:
        return default
    return raw


def _decode_private_test_cases(encoded: str) -> list[dict[str, Any]]:
    payload = str(encoded or "").strip()
    if not payload:
        return []
    try:
        loaded = json.loads(payload)
        return loaded if isinstance(loaded, list) else []
    except Exception:
        pass

    try:
        decoded = base64.b64decode(payload)
        decompressed = zlib.decompress(decoded)
        unpacked = pickle.loads(decompressed)
        loaded = json.loads(unpacked)
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("LiveCodeBench row must be a JSON object")

        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError("LiveCodeBench row missing 'id'")

        question = str(row.get("question_content", "")).strip()
        if not question:
            raise ValueError(f"Empty question_content for sample {sample_id}")

        starter_code = str(row.get("starter_code", ""))

        public_cases = _safe_json_loads(row.get("public_test_cases"), [])
        private_cases = _decode_private_test_cases(str(row.get("private_test_cases", "")))
        if not isinstance(public_cases, list):
            public_cases = []
        if not isinstance(private_cases, list):
            private_cases = []

        all_cases = list(public_cases) + list(private_cases)
        inputs = [str(case.get("input", "")) for case in all_cases if isinstance(case, dict)]
        outputs = [str(case.get("output", "")) for case in all_cases if isinstance(case, dict)]
        inputs = _ensure_str_list(inputs, "inputs", sample_id)
        outputs = _ensure_str_list(outputs, "outputs", sample_id)
        if len(inputs) != len(outputs):
            raise ValueError(
                f"inputs/outputs length mismatch for sample {sample_id}: "
                f"{len(inputs)} vs {len(outputs)}"
            )

        metadata = _safe_json_loads(row.get("metadata"), {})
        if not isinstance(metadata, dict):
            metadata = {}

        fn_name_raw = metadata.get("func_name")
        fn_name = str(fn_name_raw).strip() if fn_name_raw is not None else None
        if fn_name == "":
            fn_name = None

        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                meta={
                    "question_id": str(row.get("question_id", sample_id)).strip(),
                    "platform": str(row.get("platform", "")).strip(),
                    "difficulty": str(row.get("difficulty", "")).strip(),
                    "contest_id": str(row.get("contest_id", "")).strip(),
                    "contest_date": str(row.get("contest_date", "")).strip(),
                    "source_subset": str(row.get("source_subset", "")).strip(),
                },
                data={
                    "question_content": question,
                    "starter_code": starter_code,
                    "fn_name": fn_name,
                    "inputs": inputs,
                    "outputs": outputs,
                    "num_public_tests": len(public_cases),
                    "num_private_tests": len(private_cases),
                    "timeout_sec": int(row.get("timeout_sec", 6)),
                },
            )
        )

    return samples


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    question = str(sample.data["question_content"])
    starter_code = str(sample.data.get("starter_code", "")).rstrip()

    user_prompt = (
        "You will be given a programming problem. "
        "Write a correct Python program that matches the specification and passes all tests.\n\n"
        f"Question:\n{question}\n\n"
    )

    if starter_code:
        user_prompt += (
            "Use the following starter code and complete it.\n"
            "Return only Python code enclosed in triple backticks.\n\n"
            f"```python\n{starter_code}\n```\n"
        )
    else:
        user_prompt += (
            "Read from stdin and write to stdout.\n"
            "Return only Python code enclosed in triple backticks.\n\n"
            "```python\n# YOUR CODE HERE\n```\n"
        )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
