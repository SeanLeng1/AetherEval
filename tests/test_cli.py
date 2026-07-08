import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from aethereval.cli import _build_external_spec, build_parser, run_external_benchmark


class ExternalCliTests(unittest.TestCase):
    def test_bfcl_external_spec(self) -> None:
        args = build_parser().parse_args(
            [
                "--external-benchmark",
                "bfcl",
                "--model",
                "rlla-gdpo",
                "--output-dir",
                "outputs/bfcl",
                "--categories",
                "non_live,live",
                "--tp-size",
                "4",
                "--temperature",
                "0.2",
                "--no-overwrite",
                "--skip-evaluation",
            ]
        )

        spec, _run = _build_external_spec(args)

        self.assertEqual(spec.categories, ["non_live", "live"])
        self.assertEqual(spec.num_gpus, 4)
        self.assertEqual(spec.temperature, 0.2)
        self.assertFalse(spec.allow_overwrite)
        self.assertTrue(spec.run_generation)
        self.assertFalse(spec.run_evaluation)

    def test_external_requires_output_dir(self) -> None:
        args = build_parser().parse_args(
            ["--external-benchmark", "bfcl", "--model", "rlla-gdpo"]
        )

        with self.assertRaisesRegex(ValueError, "--output-dir"):
            _build_external_spec(args)

    def test_bfcl_skip_only_does_not_require_bfcl_eval(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "bfcl"
            args = build_parser().parse_args(
                [
                    "--external-benchmark",
                    "bfcl",
                    "--model",
                    "dry-model",
                    "--output-dir",
                    str(out),
                    "--skip-generation",
                    "--skip-evaluation",
                ]
            )

            result = run_external_benchmark(args)

            self.assertEqual(result["metrics"], {})
            self.assertEqual(result["primary_metric"], "avg_acc")
            self.assertEqual(result["primary_score"], 0.0)
            self.assertTrue((out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
