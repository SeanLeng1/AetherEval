import tempfile
import unittest
from pathlib import Path

from aethereval.core.primary_score import (
    normalize_task_summary,
    primary_score_fields,
)
from aethereval.core.run_summary import build_run_summary


class PrimaryScoreTests(unittest.TestCase):
    def summary(self, items, selected=None):
        with tempfile.TemporaryDirectory() as tmp:
            return build_run_summary(
                run_root=Path(tmp), run_id="test", selected_tasks=selected or list(items),
                model="model", model_name="model", backend="test",
                phase="generate_and_eval", task_summaries=items,
            )

    def test_mixed_units_use_percentages_and_keep_native_metrics(self):
        items = {
            "math": {"primary_metric": "accuracy", "primary_score": 0.8,
                     "metrics": {"accuracy": 0.8}, "evaluation_complete": True},
            "writing": {"primary_metric": "overall_score", "primary_score": 80.0,
                        "metrics": {"overall_score": 80.0}, "evaluation_complete": True},
            "bfcl": {"primary_metric": "overall_acc", "primary_score": 0.8,
                     "metrics": {"overall_acc": 0.8}, "prediction_records": 10,
                     "prediction_scored_records": 10},
        }
        summary = self.summary(items)
        self.assertEqual(summary["primary_score_unit"], "percent")
        self.assertEqual(summary["primary_score_aggregate"], (80 + 80 + 0.8) / 3)
        self.assertEqual(summary["results"]["math"]["metrics"]["accuracy"], 0.8)
        self.assertEqual(summary["primary_scores"]["math"]["raw_score"], 0.8)
        # A percentage below one is still a percentage; never infer units from magnitude.
        self.assertEqual(summary["primary_scores"]["bfcl"]["score"], 0.8)
        self.assertEqual(items["math"]["primary_score"], 0.8)

    def test_normalizing_a_saved_summary_does_not_scale_twice(self):
        item = {"primary_metric": "accuracy", "metrics": {"accuracy": 0.75},
                "primary_score": 0.75, "evaluation_complete": True}
        normalized = normalize_task_summary(item)
        self.assertEqual(normalized["primary_score"], 75.0)
        self.assertEqual(normalize_task_summary(normalized), normalized)

    def test_unbounded_rewards_are_preserved_and_excluded_from_the_mean(self):
        summary = self.summary({
            "math": {"primary_metric": "accuracy", "primary_score": 0.8,
                     "metrics": {"accuracy": 0.8}, "evaluation_complete": True},
            "reward": {"primary_metric": "overall/reward", "primary_score": -1.5,
                       "metrics": {"overall/reward": -1.5}, "evaluation_complete": True},
        })
        self.assertEqual(summary["primary_score_aggregate"], 80.0)
        self.assertEqual(summary["primary_score_aggregate_tasks"], ["math"])
        self.assertEqual(summary["primary_score_excluded_tasks"], ["reward"])
        reward = summary["results"]["reward"]
        self.assertEqual(reward["raw_primary_score"], -1.5)
        self.assertIsNone(reward["primary_score"])

    def test_partial_or_missing_tasks_do_not_produce_a_complete_run_score(self):
        complete = {"primary_metric": "accuracy", "metrics": {"accuracy": 1.0},
                    "evaluation_complete": True}
        for incomplete in (
            {**complete, "evaluation_complete": False},
            {"primary_metric": "overall_acc", "metrics": {"overall_acc": 100.0},
             "prediction_records": 2, "prediction_scored_records": 1},
            {**complete, "metrics": {}, "primary_score": None},
        ):
            with self.subTest(incomplete=incomplete):
                self.assertIsNone(self.summary({"task": incomplete})["primary_score_aggregate"])
        self.assertIsNone(self.summary({"task": complete}, ["task", "missing"])["primary_score_aggregate"])

    def test_custom_metric_can_declare_its_units(self):
        fields = primary_score_fields("custom", 7.5, scale=10.0)
        self.assertEqual(fields["raw_primary_score"], 7.5)
        self.assertEqual(fields["primary_score"], 75.0)
        self.assertIsNone(primary_score_fields("custom", 7.5, scale=None)["primary_score"])

    def test_invalid_units_and_scores_fail_instead_of_silently_clipping(self):
        for raw, scale in ((1.1, 100), (-0.1, 100), (float("nan"), 100),
                           (1, 0), (1, float("inf")), (1, "100")):
            with self.subTest(raw=raw, scale=scale), self.assertRaises(ValueError):
                primary_score_fields("accuracy", raw, scale=scale)


if __name__ == "__main__":
    unittest.main()
