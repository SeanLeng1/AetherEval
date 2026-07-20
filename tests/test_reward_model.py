import unittest
from unittest import mock

from benchmark_utils import reward_model


class _FakeService:
    instances = []

    def __init__(self, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs
        self.calls = []
        self.closed = False
        self.index = len(self.instances)
        self.instances.append(self)

    def request_many(self, path, payloads, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append((path, list(payloads), dict(kwargs)))
        return [
            {"data": [{"embedding": [float(self.index * 10 + index)]}]}
            for index in range(len(payloads))
        ]

    def close(self):
        self.closed = True


class RewardModelTests(unittest.TestCase):
    def test_sglang_backend_scores_models_sequentially_with_raw_logits(self) -> None:
        _FakeService.instances = []
        conversations = [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ]
        with (
            mock.patch.object(reward_model, "SGLangService", _FakeService),
            mock.patch.object(
                reward_model,
                "_render_conversations",
                return_value=["rendered a", "rendered b"],
            ) as render,
        ):
            backend = reward_model.SGLangRewardModelBackend(
                dp_size=4,
                tensor_parallel_size=2,
            )
            scores = backend.score_reward_models(
                ["rm", "cm"],
                conversations,
                {
                    "max_length": 1024,
                    "dtype": "bfloat16",
                    "trust_remote_code": True,
                    "sglang_args": {"mem_fraction_static": 0.8},
                },
            )

        self.assertEqual(scores, {"rm": [0.0, 1.0], "cm": [10.0, 11.0]})
        self.assertEqual(len(_FakeService.instances), 2)
        self.assertTrue(all(service.closed for service in _FakeService.instances))
        for model, service in zip(("rm", "cm"), _FakeService.instances, strict=True):
            self.assertEqual(service.kwargs["model"], model)
            self.assertEqual(service.kwargs["dp_size"], 4)
            self.assertEqual(service.kwargs["tensor_parallel_size"], 2)
            self.assertEqual(service.kwargs["model_kwargs"]["dtype"], "bfloat16")
            self.assertEqual(
                service.kwargs["model_kwargs"]["cuda_graph_backend_decode"],
                "disabled",
            )
            self.assertEqual(
                service.kwargs["model_kwargs"]["cuda_graph_backend_prefill"],
                "disabled",
            )
            self.assertIs(service.kwargs["model_kwargs"]["is_embedding"], True)
            self.assertEqual(
                service.kwargs["model_kwargs"]["mem_fraction_static"], 0.8
            )
            self.assertEqual(service.calls[0][0], "/v1/embeddings")
            self.assertEqual(service.calls[0][1][0]["input"], "rendered a")
        self.assertEqual(render.call_count, 2)

    def test_scalar_embedding_requires_one_raw_score(self) -> None:
        self.assertEqual(
            reward_model._extract_scalar_embedding(
                {"data": [{"embedding": [-1.25]}]}
            ),
            -1.25,
        )
        with self.assertRaisesRegex(ValueError, "exactly one raw score"):
            reward_model._extract_scalar_embedding(
                {"data": [{"embedding": [0.1, 0.9]}]}
            )


if __name__ == "__main__":
    unittest.main()
