import concurrent.futures
import unittest

from aethereval.core.types import GenerationOutput
from benchmark_utils.llm_judge import chat_completion, resolve_judge_settings
from benchmark_utils.local_judge import OfflineJudgeClient


class _FakeBackend:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def generate(self, inputs, gen_cfg):  # noqa: ANN001
        self.calls.append((list(inputs), dict(gen_cfg)))
        return [
            GenerationOutput(
                sample_id=item.sample_id,
                prompt=item.prompt,
                generations=[f"judged:{item.prompt[-1]['content']}"],
                meta={"prompt_token_count": 1, "response_token_counts": [1]},
            )
            for item in inputs
        ]

    def close(self) -> None:
        self.closed = True


class _BackendSystemValidator(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.validation_calls = []

    def validate_system_role_support(self, chat_template_kwargs):  # noqa: ANN001
        self.validation_calls.append(dict(chat_template_kwargs))


class _RecordingClient:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append((messages, kwargs))
        return "local-result"


class _SystemAwareTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        **kwargs,  # noqa: ANN001
    ):
        del kwargs
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(message["content"] for message in messages)


class _SystemRejectingTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        **kwargs,  # noqa: ANN001
    ):
        del messages, tokenize, add_generation_prompt, kwargs
        raise ValueError("System role not supported")


class _SystemDroppingTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        **kwargs,  # noqa: ANN001
    ):
        del tokenize, add_generation_prompt, kwargs
        return messages[-1]["content"]


class OfflineJudgeTests(unittest.TestCase):
    def test_concurrent_requests_are_submitted_as_one_offline_batch(self) -> None:
        backend = _FakeBackend()
        client = OfflineJudgeClient(
            model="local/judge",
            dp_size=1,
            tensor_parallel_size=1,
            batch_size=8,
            batch_wait_seconds=0.05,
            backend=backend,
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        client.complete,
                        [{"role": "user", "content": f"prompt-{index}"}],
                        temperature=0.5,
                        max_tokens=128,
                    )
                    for index in range(4)
                ]
                outputs = [future.result() for future in futures]
        finally:
            client.close()

        self.assertEqual(len(backend.calls), 1)
        self.assertCountEqual(outputs, [f"judged:prompt-{index}" for index in range(4)])
        self.assertEqual(backend.calls[0][1]["temperature"], 0.5)
        self.assertEqual(backend.calls[0][1]["max_new_tokens"], 128)
        self.assertIs(backend.calls[0][1]["enable_thinking"], False)
        self.assertIs(backend.calls[0][1]["_show_progress"], False)
        self.assertTrue(backend.closed)

    def test_system_role_is_preflighted_before_offline_generation(self) -> None:
        backend = _FakeBackend()
        client = OfflineJudgeClient(
            model="local/judge",
            dp_size=1,
            tensor_parallel_size=1,
            backend=backend,
            tokenizer=_SystemAwareTokenizer(),
        )
        try:
            output = client.complete(
                [
                    {"role": "system", "content": "grade carefully"},
                    {"role": "user", "content": "grade"},
                ]
            )
        finally:
            client.close()

        self.assertEqual(output, "judged:grade")
        self.assertEqual(len(backend.calls), 1)

    def test_system_role_preflight_can_reuse_backend_worker_tokenizer(self) -> None:
        backend = _BackendSystemValidator()
        client = OfflineJudgeClient(
            model="local/judge",
            dp_size=8,
            tensor_parallel_size=1,
            backend=backend,
        )
        try:
            client.complete(
                [
                    {"role": "system", "content": "grade carefully"},
                    {"role": "user", "content": "grade"},
                ]
            )
        finally:
            client.close()

        self.assertEqual(backend.validation_calls, [{"enable_thinking": False}])

    def test_rejected_system_role_has_clear_error(self) -> None:
        backend = _FakeBackend()
        client = OfflineJudgeClient(
            model="local/gemma",
            dp_size=1,
            tensor_parallel_size=1,
            backend=backend,
            tokenizer=_SystemRejectingTokenizer(),
        )
        try:
            with self.assertRaisesRegex(
                ValueError,
                "local/gemma.*does not support system-role.*System role not supported",
            ):
                client.complete(
                    [
                        {"role": "system", "content": "grade carefully"},
                        {"role": "user", "content": "grade"},
                    ]
                )
        finally:
            client.close()

        self.assertEqual(backend.calls, [])

    def test_silently_dropped_system_role_has_clear_error(self) -> None:
        backend = _FakeBackend()
        client = OfflineJudgeClient(
            model="local/judge",
            dp_size=1,
            tensor_parallel_size=1,
            backend=backend,
            tokenizer=_SystemDroppingTokenizer(),
        )
        try:
            with self.assertRaisesRegex(ValueError, "does not preserve system-role"):
                client.complete(
                    [
                        {"role": "system", "content": "grade carefully"},
                        {"role": "user", "content": "grade"},
                    ]
                )
        finally:
            client.close()

        self.assertEqual(backend.calls, [])

    def test_shared_chat_completion_routes_to_offline_client_without_api_key(
        self,
    ) -> None:
        client = _RecordingClient()
        settings = resolve_judge_settings(
            {
                "judge_model": "local/judge",
                "judge_temperature": 0.25,
                "judge_max_new_tokens": 96,
                "judge_top_p": 0.8,
                "judge_enable_thinking": False,
                "_judge_client": client,
            },
            default_model="unused",
        )

        output = chat_completion(
            settings,
            [{"role": "user", "content": "grade"}],
        )

        self.assertEqual(output, "local-result")
        self.assertEqual(settings.base_url, "offline://local-judge")
        self.assertEqual(client.calls[0][1]["temperature"], 0.25)
        self.assertEqual(client.calls[0][1]["max_tokens"], 96)
        self.assertEqual(client.calls[0][1]["top_p"], 0.8)
        self.assertEqual(client.calls[0][1]["extra_body"], {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
