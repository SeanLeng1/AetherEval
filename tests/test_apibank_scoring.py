import unittest
from pathlib import Path

from aethereval.core.types import Sample
from benchmarks.apibank.metrics import (
    PRIMARY_METRIC,
    aggregate as aggregate_native,
    score_generation as score_generation_native,
)
from benchmarks.apibank.scoring import (
    aggregate_scores,
    compute_correctness_score,
    compute_length_score,
    extract_tool_calls_lenient,
    parse_assistant_output,
    score_record,
    validate_output_format,
)
from benchmarks.apibank.task import build_prompt, load_samples

ANSWER = [{"name": "SymptomSearch", "parameters": {"symptom": "rash"}}]

RAW_MATCH = (
    "<think>check the symptom database</think>\n"
    '<tool_call>\n{"name": "SymptomSearch", "parameters": {"symptom": "rash"}}\n</tool_call>'
)
RAW_WRONG_PARAMS = (
    "<think>check the symptom database</think>\n"
    '<tool_call>\n{"name": "SymptomSearch", "parameters": {"symptom": "fever"}}\n</tool_call>'
)
RAW_MISSING_TAGS = '{"name": "SymptomSearch", "parameters": {"symptom": "rash"}}'


def _scored(raw: str, answer=ANSWER) -> dict:
    thought, tool_calls = parse_assistant_output(raw)
    return score_record(
        {
            "data": {"answer": answer},
            "raw_output": raw,
            "thought": thought,
            "tool_calls": tool_calls,
        }
    )


class ApiBankScoringTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        record = _scored(RAW_MATCH)
        self.assertEqual(record["score"], 1)
        self.assertEqual(record["loose_score"], 1)
        self.assertEqual(record["format_score"], 1)
        self.assertEqual(record["format_errors"], [])
        self.assertEqual(record["length_score"], 0.01)  # round(4 / 512, 2)
        self.assertEqual(record["think_word_count"], 4)

    def test_wrong_params(self) -> None:
        record = _scored(RAW_WRONG_PARAMS)
        self.assertEqual(record["score"], 0)
        self.assertEqual(record["loose_score"], 0)  # wrong params fail both parsers
        self.assertEqual(record["format_score"], 1)
        self.assertEqual(record["length_score"], 0.01)

    def test_missing_tags(self) -> None:
        record = _scored(RAW_MISSING_TAGS)
        self.assertEqual(record["score"], 0)  # strict: no tags -> dropped
        self.assertEqual(record["loose_score"], 1)  # loose: bare JSON recovered
        self.assertEqual(record["format_score"], 0)
        self.assertEqual(record["format_errors"], ["missing_think"])
        self.assertEqual(record["length_score"], 0.0)
        self.assertEqual(record["think_word_count"], 0)

    def test_lenient_extraction(self) -> None:
        # ToolRL parser: last <tool_call> block, no closing-tag guard.
        # Unclosed last block -> strict drops it, loose (ToolRL) recovers.
        unclosed = (
            "<think>t</think>\n<tool_call>\n"
            '{"name": "SymptomSearch", "parameters": {"symptom": "rash"}}'
        )
        self.assertEqual(parse_assistant_output(unclosed)[1], [])  # strict: no </tool_call>
        self.assertEqual(
            compute_correctness_score(extract_tool_calls_lenient(unclosed), ANSWER), 1
        )
        # Correct call in a NON-last block is still missed (ToolRL takes only the last).
        non_last = (
            "<think>t</think>\n<tool_call>\n"
            '{"name": "SymptomSearch", "parameters": {"symptom": "rash"}}\n'
            "<tool_call>\n<response>done</response>"
        )
        self.assertEqual(extract_tool_calls_lenient(non_last), [])  # last block has no JSON
        # Bare JSON with no tags at all -> whole output is the block -> recovered.
        self.assertEqual(
            extract_tool_calls_lenient(RAW_MISSING_TAGS),
            [{"name": "SymptomSearch", "parameters": {"symptom": "rash"}}],
        )

    def test_correctness_variants(self) -> None:
        # Bare parameter dict (no name/parameters keys) inherits the gold name.
        self.assertEqual(compute_correctness_score([{"symptom": "rash"}], ANSWER), 1)
        # Gold answer as plain dict instead of single-element list.
        self.assertEqual(compute_correctness_score([ANSWER[0]], ANSWER[0]), 1)
        # String tool call is json-parsed.
        self.assertEqual(
            compute_correctness_score(
                ['{"name": "SymptomSearch", "parameters": {"symptom": "rash"}}'], ANSWER
            ),
            1,
        )
        # Malformed gold answer -> None (excluded from acc).
        self.assertIsNone(compute_correctness_score([ANSWER[0]], "not-a-dict"))
        # Malformed string tool call aborts matching -> 0.
        self.assertEqual(compute_correctness_score(["{bad json", ANSWER[0]], ANSWER), 0)
        # Reference script treats non-dict containers as non-matches and keeps scanning.
        self.assertEqual(compute_correctness_score([[], ANSWER[0]], ANSWER), 1)

    def test_format_variants(self) -> None:
        self.assertEqual(
            validate_output_format("<think>t</think>\n<response>r</response>"), (1, [])
        )
        self.assertEqual(
            validate_output_format("<think>t</think>"),
            (0, ["missing_tool_call_and_response"]),
        )
        self.assertEqual(
            validate_output_format("<think>t\n<tool_call>x</tool_call>"),
            (0, ["unbalanced_think_tags"]),
        )
        self.assertEqual(validate_output_format(""), (0, ["empty_output"]))
        self.assertEqual(
            validate_output_format(
                "<think>t</think><response>r</response><tool_call>x</tool_call>"
            ),
            (0, ["response_before_tool_call_end", "tool_call_after_response"]),
        )

    def test_length_cap(self) -> None:
        raw = "<think>" + " ".join(["w"] * 600) + "</think>\n<response>r</response>"
        self.assertEqual(compute_length_score(raw), (1.0, 600))

    def test_aggregate(self) -> None:
        scores = {
            "Level1_0": _scored(RAW_MATCH),
            "Level1_1": _scored(RAW_WRONG_PARAMS),
            "Level2_0": _scored(RAW_MISSING_TAGS),
        }
        record = aggregate_scores(scores)
        self.assertEqual(record["correct_lv1"], 1)
        self.assertEqual(record["total_lv1"], 2)
        self.assertEqual(record["lv1_acc"], 50.0)
        self.assertEqual(record["lv2_acc"], 0.0)
        self.assertIsNone(record["lv3_acc"])
        self.assertEqual(record["overall_acc"], 33.33)
        # Loose recovers Level2_0 (bare-JSON) that strict dropped: overall 33.33 -> 66.67.
        self.assertEqual(record["loose_lv1_acc"], 50.0)
        self.assertEqual(record["loose_lv2_acc"], 100.0)
        self.assertIsNone(record["loose_lv3_acc"])
        self.assertEqual(record["loose_overall_acc"], 66.67)
        self.assertEqual(record["LooseCorrectAcc."], 66.67)
        self.assertEqual(record["Loose Level 2 Acc."], 100.0)
        self.assertEqual(record["format_lv1_acc"], 100.0)
        self.assertEqual(record["format_lv2_acc"], 0.0)
        self.assertEqual(record["overall_format_acc"], 66.67)
        self.assertEqual(record["length_avg_lv1"], 0.01)
        self.assertEqual(record["length_avg_lv2"], 0.0)
        self.assertEqual(record["overall_length_avg"], 0.0067)
        self.assertEqual(record["Level 1 Acc."], 50.0)
        self.assertEqual(record["Level 2 Acc."], 0.0)
        self.assertIsNone(record["Level 3 Acc."])
        self.assertEqual(record["lv1_format"], 100.0)
        self.assertEqual(record["overall_format"], 66.67)
        self.assertEqual(record["lv1_length"], 0.01)
        self.assertEqual(record["overall_length"], 0.0067)
        self.assertEqual(record["CorrectAcc."], 33.33)
        self.assertEqual(record["FormatAcc."], 66.67)
        self.assertEqual(record["LengthRew."], 0.0067)
        self.assertEqual(record["LengthReward"], 0.0067)
        self.assertEqual(record["Overall"], 1.0067)
        self.assertEqual(record["think_word_count_avg_lv1"], 4.0)
        self.assertEqual(record["overall_think_word_count_avg"], 2.6667)
        self.assertEqual(record["reward_avg_lv1"], 0.01)
        self.assertEqual(record["overall_reward_avg"], 0.0067)

    def test_native_task_loads_jsonl(self) -> None:
        task_dir = Path(__file__).resolve().parents[1] / "benchmarks" / "apibank"
        samples = load_samples(task_dir)
        self.assertEqual(len(samples), 597)
        self.assertEqual(samples[0].id, "Level1_0")
        self.assertEqual(samples[0].meta["level"], 1)
        prompt = build_prompt(samples[0])
        self.assertEqual([message["role"] for message in prompt], ["system", "user"])
        self.assertTrue(prompt[0]["content"])
        self.assertTrue(prompt[1]["content"])

    def test_native_metrics_path(self) -> None:
        sample = Sample(id="Level1_0", gold=ANSWER, meta={"level": 1})
        scored = score_generation_native(sample, RAW_MATCH)
        record = {
            "sample_id": sample.id,
            "gen_idx": 0,
            "prompt": [],
            "generation": RAW_MATCH,
            "score": scored["score"],
            "is_pass": scored["is_pass"],
            "parsed": scored["parsed"],
            "gold": sample.gold,
            "error": None,
            "meta": scored["meta"],
        }
        metrics = aggregate_native(
            [
                {
                    "sample_id": sample.id,
                    "gold": sample.gold,
                    "meta": sample.meta,
                    "scores": [scored["score"]],
                    "passes": [scored["is_pass"]],
                    "records": [record],
                }
            ],
            {"n": 1},
        )

        self.assertEqual(PRIMARY_METRIC, "Overall")
        self.assertEqual(metrics["CorrectAcc."], 100.0)
        self.assertEqual(metrics["FormatAcc."], 100.0)
        self.assertEqual(metrics["LengthRew."], 0.01)
        self.assertEqual(metrics["Overall"], 2.01)


if __name__ == "__main__":
    unittest.main()
