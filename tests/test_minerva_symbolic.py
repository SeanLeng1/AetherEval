import unittest
from pathlib import Path

from aethereval.core.task_register import load_task


class MinervaSymbolicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle = load_task("minerva")
        task_dir = Path(__file__).resolve().parents[1] / "benchmarks" / "minerva"
        cls.samples = {sample.id: sample for sample in bundle.task_module.load_samples(task_dir)}
        cls.score = staticmethod(bundle.metrics_module.score_generation)

    def test_correct_equivalent_answers_for_all_five_physics_questions(self):
        cases = {
            27: (
                r"\frac{dM}{dt}=\frac{10^{8}L_{\odot}M^{6}}{7c^{2}M_{\odot}^{6}}",
                r"\frac{10^{8}L_{\odot}M^{6}}{7c^{2}M_{\odot}^{6}}",
                r"7c^{2}M_{\odot}^{6}\frac{dM}{dt}=10^{8}L_{\odot}M^{6}",
            ),
            138: (
                r"m_p c^2(\gamma^2-1)(1-\cos^2\theta)",
                r"(\gamma^2-1)m_p c^2\sin^2\theta",
            ),
            261: (
                r"\hbar\omega(v+\frac12)-\frac{(eE_0)^2}{2m\omega^2}",
                r"-\frac{e^2 E_0^2}{2m\omega^2}+\hbar\omega(v+\frac12)",
            ),
            268: (r"\frac{E_1+2E_2}{3}", r"\frac{2E_2}{3}+\frac{E_1}{3}"),
            269: (r"E_2,E_1", r"\{E_2,E_1\}"),
        }
        for index, answers in cases.items():
            sample = self.samples[f"minervamath_{index}"]
            for answer in answers:
                with self.subTest(index=index, answer=answer):
                    result = self.score(sample, r"\boxed{" + answer + "}")
                    self.assertEqual(result["score"], 1.0, result)

    def test_wrong_and_nonfinite_answers_are_rejected(self):
        cases = {
            27: (
                r"\frac{dM}{dt}=\frac{10^5 L_{\odot}M^6}{c^2M_{\odot}^6}",
                r"\frac{dN}{dt}=\frac{10^8 L_{\odot}M^6}{7c^2M_{\odot}^6}",
                r"0",
            ),
            138: (r"m_p c^2(\gamma^2+1)\sin^2\theta", r"\infty", r"0"),
            261: (
                r"\omega",
                r"\hbar\omega(v+\frac12)+\frac{e^2 E_0^2}{2m\omega^2}",
                r"\hbar\omega(v+\frac12)-\frac{\exp(2) E_0^2}{2m\omega^2}",
                r"\hbar\omega(v+\frac12)",
            ),
            268: (r"\frac{2E_1+E_2}{3}", r"E_1"),
            269: (r"E_1", r"E_1,E_2,E_3"),
        }
        for index, answers in cases.items():
            sample = self.samples[f"minervamath_{index}"]
            for answer in answers:
                with self.subTest(index=index, answer=answer):
                    result = self.score(sample, r"\boxed{" + answer + "}")
                    self.assertEqual(result["score"], 0.0, result)

    def test_prediction_uses_the_final_box_not_an_earlier_correct_expression(self):
        sample = self.samples["minervamath_261"]
        prediction = (
            r"Initially \boxed{\hbar\omega(v+\frac12)-\frac{e^2 E_0^2}{2m\omega^2}}"
            r" but my final answer is \boxed{\omega}."
        )
        self.assertEqual(self.score(sample, prediction)["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
