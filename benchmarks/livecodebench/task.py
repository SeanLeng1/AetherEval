import base64
import binascii
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample
from benchmark_utils.code_prompt import build_code_prompt


TASK_NAME = "livecodebench"
DATA_FILE = "data/eval.jsonl"


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


def _json_loads_field(raw: Any, key: str, sample_id: str) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} is invalid JSON for sample {sample_id}") from exc
    return raw


def _decode_private_test_cases(encoded: str, sample_id: str) -> list[dict[str, Any]]:
    payload = str(encoded or "").strip()
    if not payload:
        return []
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError:
        try:
            decoded = base64.b64decode(payload, validate=True)
            decompressed = zlib.decompress(decoded)
            unpacked = pickle.loads(decompressed)
            loaded = json.loads(unpacked)
        except (
            binascii.Error,
            EOFError,
            json.JSONDecodeError,
            pickle.UnpicklingError,
            TypeError,
            ValueError,
            zlib.error,
        ) as exc:
            raise ValueError(
                f"private_test_cases is invalid for sample {sample_id}"
            ) from exc

    if not isinstance(loaded, list):
        raise ValueError(
            f"private_test_cases must decode to a list for sample {sample_id}"
        )
    return loaded


def _case_field(case: dict[str, Any], key: str, sample_id: str, idx: int) -> str:
    value = str(case[key])
    if value == "":
        raise ValueError(f"test_cases[{idx}].{key} is empty for sample {sample_id}")
    return value


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("LiveCodeBench row must be a JSON object")

        sample_id = str(row["id"]).strip()
        if not sample_id:
            raise ValueError("LiveCodeBench row missing 'id'")

        question = str(row["question_content"]).strip()
        if not question:
            raise ValueError(f"Empty question_content for sample {sample_id}")

        starter_code = str(row.get("starter_code", ""))

        public_cases = _json_loads_field(
            row["public_test_cases"], "public_test_cases", sample_id
        )
        private_cases = _decode_private_test_cases(
            str(row["private_test_cases"]), sample_id
        )
        if not isinstance(public_cases, list):
            raise ValueError(f"public_test_cases must be a list for sample {sample_id}")

        all_cases = list(public_cases) + list(private_cases)
        inputs: list[str] = []
        outputs: list[str] = []
        for idx, case in enumerate(all_cases):
            if not isinstance(case, dict):
                raise ValueError(
                    f"test_cases[{idx}] must be an object for sample {sample_id}"
                )
            inputs.append(_case_field(case, "input", sample_id, idx))
            outputs.append(_case_field(case, "output", sample_id, idx))
        inputs = _ensure_str_list(inputs, "inputs", sample_id)
        outputs = _ensure_str_list(outputs, "outputs", sample_id)
        if len(inputs) != len(outputs):
            raise ValueError(
                f"inputs/outputs length mismatch for sample {sample_id}: "
                f"{len(inputs)} vs {len(outputs)}"
            )

        metadata = _json_loads_field(row["metadata"], "metadata", sample_id)
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be a JSON object for sample {sample_id}")

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
    # AetherRL's code prompt. Functional problems show their starter code in place of
    # the placeholder, as the official template does, so the class and signature are given.
    question = str(sample.data["question_content"])
    starter_code = str(sample.data.get("starter_code", ""))
    if not starter_code:
        return build_code_prompt(question, stdin=True)
    return build_code_prompt(
        question, fn_name=sample.data.get("fn_name"), code_template=starter_code
    )
