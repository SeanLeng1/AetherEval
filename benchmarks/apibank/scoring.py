"""Faithful port of the GD2PO API-Bank scorer (``evaluate_reward.py``).

Per-sample scores:
  - correctness: exact tool-name + parameter-dict match against the gold answer
    (``None`` when the gold answer itself is malformed -> excluded from acc).
  - format: ``<think>``/``<tool_call>``/``<response>`` tag-structure check.
  - length: ``round(think_word_count / 512, 2)`` capped at 1.0.

``aggregate_scores`` keeps the reference per-level record schema and adds the
APIBank report columns used by AetherEval.
"""

import json
from typing import Any

LEVELS = ("lv1", "lv2", "lv3")

# Reference eval-time length constants (env overrides are never set by the eval pipeline).
MAX_REWARD_LEN = 512


def normalize_answer(answer: Any) -> Any:
    if isinstance(answer, list) and answer:
        return answer[0]
    return answer


def compute_correctness_score(tool_calls: list[Any], answer: Any) -> int | None:
    answer = normalize_answer(answer)
    if not isinstance(answer, dict):
        return None

    answer_name = answer.get("name")
    answer_parameters = answer.get("parameters")
    if not isinstance(answer_name, str) or not isinstance(answer_parameters, dict):
        return None

    for tool_call in tool_calls:
        if isinstance(tool_call, str):
            try:
                tool_call = json.loads(tool_call)
            except json.JSONDecodeError:
                return 0
        predict = tool_call

        try:
            has_name = "name" in predict
            has_parameters = "parameters" in predict
        except TypeError:
            return 0
        if not has_name or not has_parameters:
            name = answer_name
            parameters = predict
        else:
            name = predict["name"]
            parameters = predict["parameters"]

        if name == answer_name and parameters == answer_parameters:
            return 1

    return 0


def validate_output_format(raw_output: str) -> tuple[int, list[str]]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return 0, ["empty_output"]

    text = raw_output.strip()
    errors = []

    for tag_name in ("think", "tool_call", "response"):
        open_count = text.count(f"<{tag_name}>")
        close_count = text.count(f"</{tag_name}>")
        if open_count != close_count:
            errors.append(f"unbalanced_{tag_name}_tags")
        if open_count > 1:
            errors.append(f"multiple_{tag_name}_blocks")

    if errors:
        return 0, sorted(set(errors))

    think_start = text.find("<think>")
    think_end = text.find("</think>")
    if think_start < 0 or think_end < 0:
        return 0, ["missing_think"]
    if think_end <= think_start:
        return 0, ["invalid_think_order"]

    think_close_pos = think_end + len("</think>")
    tool_start = text.find("<tool_call>")
    tool_end = text.find("</tool_call>")
    response_start = text.find("<response>")
    response_end = text.find("</response>")

    has_tool_call = tool_start >= 0
    has_response = response_start >= 0

    if not has_tool_call and not has_response:
        return 0, ["missing_tool_call_and_response"]

    if has_tool_call:
        if tool_end <= tool_start:
            errors.append("invalid_tool_call_order")
        if tool_start < think_close_pos:
            errors.append("tool_call_before_think_end")

    if has_response:
        if response_end <= response_start:
            errors.append("invalid_response_order")
        if response_start < think_close_pos:
            errors.append("response_before_think_end")

    if has_tool_call and has_response and response_start < tool_end:
        errors.append("response_before_tool_call_end")

    if has_response and has_tool_call and response_end < tool_start:
        errors.append("tool_call_after_response")

    return int(len(errors) == 0), sorted(set(errors))


def extract_think_content(raw_output: str) -> str | None:
    if not isinstance(raw_output, str):
        return None

    think_start = raw_output.find("<think>")
    think_end = raw_output.find("</think>")
    if think_start < 0 or think_end < 0 or think_end <= think_start:
        return None

    think_start += len("<think>")
    return raw_output[think_start:think_end].strip()


def compute_length_score(raw_output: str) -> tuple[float, int]:
    think_content = extract_think_content(raw_output)
    if think_content is None:
        return 0.0, 0

    reward = round(len(think_content.split()) / MAX_REWARD_LEN, 2)
    if reward > 1.0:
        reward = 1.0
    return reward, len(think_content.split())


def parse_assistant_output(assistant_output: str) -> tuple[str, list[Any]]:
    """Reference ``generate_deterministic.py`` parse: think text + per-line tool-call JSON."""
    thought = ""
    tool_calls = []

    if "<think>" in assistant_output and "</think>" in assistant_output:
        thought = (
            assistant_output.split("<think>", 1)[-1].split("</think>", 1)[0].strip()
        )

    if "<tool_call>" in assistant_output and "</tool_call>" in assistant_output:
        tool_block = (
            assistant_output.split("<tool_call>", 1)[-1]
            .split("</tool_call>", 1)[0]
            .strip()
        )
        for line in tool_block.splitlines():
            line = line.strip()
            if not line:
                continue
            # Reference behavior: malformed model tool-call lines are ignored.
            try:
                tool_calls.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return thought, tool_calls


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach the ``evaluate_reward.py`` per-sample score fields to a generation record."""
    score = compute_correctness_score(record["tool_calls"], record["data"]["answer"])
    format_score, format_errors = validate_output_format(record["raw_output"])
    length_score, think_word_count = compute_length_score(record["raw_output"])

    record["score"] = score
    record["correct_score"] = score
    record["format_score"] = format_score
    record["length_score"] = length_score
    record["think_word_count"] = think_word_count
    record["format_errors"] = format_errors
    return record


def get_level_key(sample_key: str) -> str | None:
    if sample_key.startswith("Level1"):
        return "lv1"
    if sample_key.startswith("Level2"):
        return "lv2"
    if sample_key.startswith("Level3"):
        return "lv3"
    return None


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _round_or_none(value: float | None, ndigits: int) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def _add_report_metrics(record: dict[str, Any]) -> None:
    record["Level 1 Acc."] = record["lv1_acc"]
    record["Level 2 Acc."] = record["lv2_acc"]
    record["Level 3 Acc."] = record["lv3_acc"]

    record["lv1_format"] = record["format_lv1_acc"]
    record["lv2_format"] = record["format_lv2_acc"]
    record["lv3_format"] = record["format_lv3_acc"]
    record["overall_format"] = record["overall_format_acc"]

    record["lv1_length"] = record["length_avg_lv1"]
    record["lv2_length"] = record["length_avg_lv2"]
    record["lv3_length"] = record["length_avg_lv3"]
    record["overall_length"] = record["overall_length_avg"]

    record["CorrectAcc."] = record["overall_acc"]
    record["FormatAcc."] = record["overall_format_acc"]
    record["LengthRew."] = record["overall_length_avg"]
    record["LengthReward"] = record["LengthRew."]

    if (
        record["CorrectAcc."] is None
        or record["FormatAcc."] is None
        or record["LengthRew."] is None
    ):
        record["Overall"] = None
        return

    record["Overall"] = round(
        record["CorrectAcc."] / 100
        + record["FormatAcc."] / 100
        + record["LengthRew."],
        4,
    )


def aggregate_scores(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reference record plus APIBank report columns."""
    correct = {lv: 0 for lv in LEVELS}
    total_correct = {lv: 0 for lv in LEVELS}

    format_pass = {lv: 0 for lv in LEVELS}
    total_format = {lv: 0 for lv in LEVELS}

    length_sum = {lv: 0.0 for lv in LEVELS}
    length_count = {lv: 0 for lv in LEVELS}

    think_sum = {lv: 0.0 for lv in LEVELS}
    think_count = {lv: 0 for lv in LEVELS}

    reward_sum = {lv: 0.0 for lv in LEVELS}
    reward_count = {lv: 0 for lv in LEVELS}

    for key, item in scores.items():
        lv = get_level_key(key)
        if lv is None:
            continue

        total_correct[lv] += 1
        if item["score"] == 1:
            correct[lv] += 1

        total_format[lv] += 1
        if item["format_score"] == 1:
            format_pass[lv] += 1

        length_score = item["length_score"]
        if isinstance(length_score, (int, float)):
            length_sum[lv] += float(length_score)
            length_count[lv] += 1
            reward_sum[lv] += float(length_score)
            reward_count[lv] += 1

        think_word_count = item["think_word_count"]
        if isinstance(think_word_count, (int, float)):
            think_sum[lv] += float(think_word_count)
            think_count[lv] += 1

    record: dict[str, Any] = {}

    for lv in LEVELS:
        record[f"correct_{lv}"] = correct[lv]
    for lv in LEVELS:
        record[f"total_{lv}"] = total_correct[lv]
    for lv in LEVELS:
        acc = _safe_div(correct[lv], total_correct[lv])
        record[f"{lv}_acc"] = _round_or_none(
            (acc * 100) if acc is not None else None, 2
        )

    overall_acc = _safe_div(sum(correct.values()), sum(total_correct.values()))
    record["overall_acc"] = _round_or_none(
        (overall_acc * 100) if overall_acc is not None else None, 2
    )

    for lv in LEVELS:
        record[f"format_{lv}"] = format_pass[lv]
    for lv in LEVELS:
        acc = _safe_div(format_pass[lv], total_format[lv])
        record[f"format_{lv}_acc"] = _round_or_none(
            (acc * 100) if acc is not None else None, 2
        )

    overall_format = _safe_div(sum(format_pass.values()), sum(total_format.values()))
    record["overall_format_acc"] = _round_or_none(
        (overall_format * 100) if overall_format is not None else None, 2
    )

    for lv in LEVELS:
        record[f"length_avg_{lv}"] = _round_or_none(
            _safe_div(length_sum[lv], length_count[lv]), 4
        )
    record["overall_length_avg"] = _round_or_none(
        _safe_div(sum(length_sum.values()), sum(length_count.values())), 4
    )

    for lv in LEVELS:
        record[f"think_word_count_avg_{lv}"] = _round_or_none(
            _safe_div(think_sum[lv], think_count[lv]), 4
        )
    record["overall_think_word_count_avg"] = _round_or_none(
        _safe_div(sum(think_sum.values()), sum(think_count.values())), 4
    )

    for lv in LEVELS:
        record[f"reward_avg_{lv}"] = _round_or_none(
            _safe_div(reward_sum[lv], reward_count[lv]), 4
        )
    record["overall_reward_avg"] = _round_or_none(
        _safe_div(sum(reward_sum.values()), sum(reward_count.values())), 4
    )

    _add_report_metrics(record)
    return record
