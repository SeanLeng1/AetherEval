import copy
from importlib import import_module
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from aethereval.core.task_register import load_task
from aethereval.core.types import GenerationOutput

metrics = import_module("benchmarks.safe-alignment-dynamic.metrics")
task = import_module("benchmarks.safe-alignment-dynamic.task")
prepare = import_module("benchmarks.safe-alignment-dynamic.prepare_data")
COMPONENTS, select_problems = prepare.COMPONENTS, prepare.select_problems


def protocol():
    return {
        "weights": [[0, 1], [0.5, 0.5], [1, 0]],
        "score_conditioning": {
            "components": COMPONENTS,
            "calibration": {
                "cm_sign": 1,
                "max_length": 2048,
                "models": {
                    "useful": {"mean": 10, "std": 2, "repo": "rm"},
                    "harmless": {"mean": -3, "std": 4, "repo": "cm"},
                },
            },
            "target_mapping": {
                "method": "ric_p2_quantile",
                "components": [c["key"] for c in COMPONENTS],
                "low": [-2, -1],
                "high": [2, 3],
            },
        },
    }


def fixture_samples(root):
    data = root / "data"
    data.mkdir()
    (data / "protocol.json").write_text(json.dumps(protocol()))
    rows = [
        {
            "prompt_id": f"p{i}",
            "subset": "alpaca",
            "messages": [
                {"role": "user", "content": "Earlier"},
                {"role": "assistant", "content": "Earlier reply"},
                {"role": "user", "content": f"Question {i}"},
            ],
        }
        for i in range(2)
    ]
    (data / "eval.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return task.load_samples(root)


def result_rows(samples, vectors):
    rows = []
    for sample in samples:
        z = vectors[sample.meta["condition"]]
        record = {
            "sample_id": sample.id,
            "gen_idx": 0,
            "prompt": task.build_prompt(sample),
            "generation": "answer",
            "score": 0.0,
            "is_pass": False,
            "error": None,
            "meta": {
                **sample.meta,
                "helpful_z": z[0],
                "harmless_z": z[1],
                "helpful": 10 + 2 * z[0],
                "harmless": -3 + 4 * z[1],
                "scoring": {
                    "rm_model_path": "rm",
                    "cm_model_path": "cm",
                    "rm_max_length": 2048,
                },
            },
        }
        rows.append({"meta": sample.meta, "records": [record]})
    return rows


class DynamicAlignmentTests(unittest.TestCase):
    def test_local_json_loading_needs_no_hf_or_network(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(sys.modules, {"datasets": None, "huggingface_hub": None}),
            mock.patch(
                "socket.create_connection",
                side_effect=AssertionError("Unexpected network access"),
            ),
        ):
            samples = fixture_samples(Path(directory))
            self.assertEqual(len(samples), 8)
            self.assertIn(
                "Target scores:", task.build_prompt(samples[0])[-1]["content"]
            )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.samples = fixture_samples(self.root)

    def test_discovery(self):
        bundle = load_task("safe_alignment_dynamic")
        self.assertEqual(bundle.metrics_module.PRIMARY_METRIC, "overall/utility")
        self.assertEqual(bundle.task_module.DEFAULT_GEN["max_new_tokens"], 1024)
        self.assertEqual(bundle.task_module.DEFAULT_GEN["temperature"], 0.0)
        self.assertEqual(bundle.task_module.DEFAULT_GEN["n"], 1)

    def test_full_runner_generation_scoring_and_resume(self):
        from aethereval.core.runner import _run_single_task

        class Backend:
            calls = 0

            def generate(inner, inputs, gen_cfg):
                inner.calls += 1
                return [
                    GenerationOutput(
                        item.sample_id,
                        item.prompt,
                        ["answer"] * item.num_generations,
                        meta={
                            "prompt_token_count": 30,
                            "response_token_counts": [2] * item.num_generations,
                        },
                    )
                    for item in inputs
                ]

            def score_reward_models(inner, paths, conversations, options):
                return {
                    "rm": [12.0] * len(conversations),
                    "cm": [5.0] * len(conversations),
                }

        backend = Backend()
        kwargs = dict(
            task_name=task.TASK_NAME,
            task_module=task,
            metrics_module=metrics,
            task_dir=self.root,
            backend=backend,
            task_output_dir=self.root / "run",
            gen_overrides={},
            metric_options={},
            overwrite=False,
            run_config_common={},
            tokenizer_getter=lambda: None,
            generate_only=True,
            eval_only=False,
        )
        generated = _run_single_task(**kwargs)
        self.assertFalse(generated["evaluation_complete"])
        kwargs.update(generate_only=False, eval_only=True)
        first = _run_single_task(**kwargs)
        kwargs.update(eval_only=False)
        second = _run_single_task(**kwargs)
        self.assertEqual(backend.calls, 1)
        self.assertTrue(first["evaluation_complete"])
        self.assertEqual(first["primary_score"], 1.5)
        self.assertEqual(second["primary_score"], first["primary_score"])
        config = json.loads((self.root / "run/run_config.json").read_text())
        self.assertEqual(config["protocol"], protocol())
        for line in (self.root / "run/predictions.jsonl").read_text().splitlines():
            meta = json.loads(line)["meta"]
            self.assertNotIn("protocol", meta)
            self.assertNotIn("weight_grid", meta)
            self.assertEqual(meta["protocol_hash"], task.protocol_hash(protocol()))
        changed = protocol()
        changed["weights"][1] = [0.4, 0.6]
        (self.root / "data/protocol.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "Saved evaluation protocol differs"):
            _run_single_task(**kwargs)

    def test_sample_metadata_is_compact(self):
        for sample in self.samples:
            self.assertNotIn("protocol", sample.meta)
            self.assertNotIn("weight_grid", sample.meta)
            self.assertEqual(
                sample.meta["protocol_hash"], task.protocol_hash(protocol())
            )

    def test_missing_calibration_length_fails_before_scoring(self):
        sample = self.samples[0]
        del sample.data["artifact"]["calibration"]["max_length"]
        output = GenerationOutput(sample.id, task.build_prompt(sample), ["answer"])
        with self.assertRaisesRegex(KeyError, "max_length"):
            metrics.score_generations_batch([sample], [output])

    def test_wrong_prediction_protocol_is_rejected(self):
        rows = result_rows(self.samples, [[1, 1]] * 4)
        rows[0]["records"][0]["meta"]["protocol_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "different evaluation protocol"):
            metrics.aggregate(rows, {"_protocol": protocol()})

    def test_artifact_matches_train_statistics_not_test_scores(self):
        load_artifact = prepare.load_artifact

        calibration = protocol()["score_conditioning"]["calibration"]
        metadata = self.root / "scoring_metadata.json"
        metadata.write_text(json.dumps(calibration))
        train = [
            {
                "reward_useful": 10 + 2 * i,
                "reward_harmless": -3 + 4 * i,
                "sft_eligible": True,
            }
            for i in range(10)
        ] + [{"reward_useful": 10000, "reward_harmless": 10000, "sft_eligible": False}]
        with (
            mock.patch("datasets.load_dataset", return_value=train) as loader,
            mock.patch("huggingface_hub.hf_hub_download", return_value=str(metadata)),
        ):
            artifact = load_artifact(revision="frozen-test-revision")
        self.assertEqual(loader.call_args.kwargs["split"], "train")
        self.assertEqual(loader.call_args.kwargs["revision"], "frozen-test-revision")
        np.testing.assert_allclose(artifact["target_mapping"]["low"], [0.45, 0.45])
        np.testing.assert_allclose(artifact["target_mapping"]["high"], [8.55, 8.55])

    def test_artifact_can_be_read_from_rl_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        load_artifact = prepare.load_artifact

        artifact = protocol()["score_conditioning"]
        table = pa.Table.from_pylist(
            [{"extra_info": {"score_conditioning": json.dumps(artifact)}}]
        )
        pq.write_table(table, self.root / "train.parquet")
        self.assertEqual(load_artifact(self.root), artifact)

    def test_prompt_mapping_and_control(self):
        sample = self.samples[0]
        before = copy.deepcopy(sample.data)
        self.assertEqual(sample.meta["targets"], [-2, 3])
        rendered = task.build_prompt(sample)
        self.assertEqual(rendered[:-1], sample.data["prompt"][:-1])
        self.assertTrue(
            rendered[-1]["content"].endswith(
                "Target scores: helpfulness=-2.0, harmlessness=3.0"
            )
        )
        self.assertEqual(sample.data, before)
        self.assertEqual(
            task.build_prompt(self.samples[3]), self.samples[3].data["prompt"]
        )
        targets = task.targets_for_weight([0.5, 0.5], sample.data["artifact"])
        np.testing.assert_allclose(targets, [-2 + 4 / np.sqrt(2), -1 + 4 / np.sqrt(2)])

    def test_template_matches_aetherrl_when_available(self):
        import importlib.util

        path = Path("/home/jixuanl/AetherRL/aetherrl/utils/score_conditioning.py")
        if not path.exists():
            self.skipTest("Optional cross-repository contract check")
        spec = importlib.util.spec_from_file_location("rl_score_conditioning", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for sample in self.samples:
            if sample.meta["weights"] is None:
                continue
            artifact = sample.data["artifact"]
            targets = module.weights_to_targets(
                sample.meta["weights"], artifact["target_mapping"]
            )
            np.testing.assert_allclose(targets, sample.meta["targets"])
            expected = module.condition_messages(
                sample.data["prompt"], targets, artifact["components"]
            )
            self.assertEqual(task.build_prompt(sample), expected)

    def test_rm_gets_no_target_and_uses_train_scale(self):
        class Backend:
            def score_reward_models(inner, paths, conversations, options):
                self.assertEqual(paths, ["rm", "cm"])
                self.assertEqual(options["max_length"], 2048)
                self.assertNotIn("Target scores", json.dumps(conversations))
                self.assertEqual(len(conversations[0]), 4)
                return {"rm": [12.0, 14.0], "cm": [5.0, 9.0]}

        sample = self.samples[1]
        output = GenerationOutput(sample.id, task.build_prompt(sample), ["one", "two"])
        results = metrics.score_generations_batch(
            [sample], [output], {"_backend": Backend()}
        )
        self.assertEqual([r["score"] for r in results[0]], [1.5, 2.5])
        self.assertEqual(results[0][0]["meta"]["helpful"], 12.0)
        self.assertEqual(results[0][0]["meta"]["helpful_z"], 1.0)

    def test_matching_gain_and_cross_matrix(self):
        rows = result_rows(self.samples, [[0, 2], [1, 1], [2, 0], [1, 1]])
        out = metrics.aggregate(rows, {"_protocol": protocol()})
        self.assertAlmostEqual(out["overall/utility"], 5 / 3)
        self.assertAlmostEqual(out["overall/gain_vs_best_fixed_condition"], 2 / 3)
        self.assertAlmostEqual(out["overall/gain_vs_shuffled_condition"], 2 / 3)
        saved_protocol, sources = metrics.paired_arrays(rows, protocol())
        matrix = (
            np.asarray(saved_protocol["weights"])
            @ sources["alpaca"][:, :, :2].mean(0).T
        )
        self.assertEqual(matrix[0, 2], 0)
        self.assertEqual(matrix[2, 2], 2)

    def test_ignoring_targets_has_zero_matching_gain(self):
        rows = result_rows(self.samples, [[1, 2]] * 4)
        out = metrics.aggregate(rows, {"_protocol": protocol()})
        for key in out:
            if "gain_vs_" in key:
                self.assertAlmostEqual(out[key], 0)

    def test_condition_permutation_is_detected(self):
        rows = result_rows(self.samples, [[2, 0], [1, 1], [0, 2], [1, 1]])
        self.assertLess(
            metrics.aggregate(rows, {"_protocol": protocol()})[
                "overall/gain_vs_shuffled_condition"
            ],
            0,
        )

    def test_repeats_and_source_size_do_not_change_weights(self):
        rows = result_rows(self.samples, [[1, 1]] * 4)
        for row in rows:
            row["records"] *= 3
        other = copy.deepcopy(rows[:4])
        for row in other:
            row["meta"]["data_source"] = "pku-saferlhf"
            for rec in row["records"]:
                rec["meta"]["helpful_z"] = 3
                rec["meta"]["harmless_z"] = 3
        out = metrics.aggregate(rows + other, {"_protocol": protocol()})
        self.assertEqual(out["overall/utility"], 2)

    def test_incomplete_or_duplicate_sweep_rejected(self):
        rows = result_rows(self.samples, [[1, 1]] * 4)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            metrics.aggregate(rows[:-1], {"_protocol": protocol()})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            metrics.aggregate(rows + rows[:1], {"_protocol": protocol()})

    def test_selection_dedups_before_limiting_and_is_order_independent(self):
        rows = [
            {"prompt_id": str(i), "subset": "alpaca", "messages": [], "split": "test"}
            for i in range(10)
        ]
        selected = select_problems(rows + rows, limit=4, seed=42)
        self.assertEqual(selected, select_problems(rows[::-1], limit=4, seed=42))
        self.assertEqual(len(selected), 4)
        with self.assertRaisesRegex(ValueError, "held-out"):
            select_problems([{**rows[0], "split": "train"}], limit=0, seed=42)
