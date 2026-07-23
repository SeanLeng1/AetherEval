import json
import os
import unittest
from unittest import mock

from benchmarks.bfcl._compat import ensure_bfcl_importable

ensure_bfcl_importable()

from benchmarks.bfcl.handler import (  # noqa: E402
    RLLAHandler,
    _post_native_generate,
)


class BfclHandlerTests(unittest.TestCase):
    def test_v4_format_prompt_preserves_benchmark_system_instructions(self) -> None:
        handler = RLLAHandler.__new__(RLLAHandler)
        prompt = handler._format_prompt(
            [
                {"role": "system", "content": "Persistent memory instructions."},
                {"role": "user", "content": "What do you remember?"},
            ],
            [],
        )

        self.assertIn("Persistent memory instructions.", prompt)
        self.assertIn("What do you remember?", prompt)

    def test_native_generate_preserves_prompt_and_sampling(self) -> None:
        handler = RLLAHandler.__new__(RLLAHandler)
        handler.model_path_or_id = "test/model"
        handler.temperature = 0.001
        handler.max_context_length = 32768
        handler.skip_special_tokens = False
        handler.tokenizer = mock.Mock()
        handler.tokenizer.tokenize.return_value = [1, 2]
        handler._format_prompt = mock.Mock(return_value="rendered prompt")
        handler.client = mock.Mock()

        response = mock.Mock(ok=True)
        response.json.return_value = {
            "text": "generated answer",
            "meta_info": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
            },
        }
        session = mock.Mock()
        session.post.return_value = response

        with (
            mock.patch.dict(
                os.environ,
                {
                    "RLLA_BFCL_GENERATE_URL": "http://router/generate",
                    "RLLA_BFCL_MAX_TOKENS": "123",
                    "RLLA_BFCL_TOP_P": "0.9",
                    "RLLA_BFCL_TOP_K": "-1",
                    "RLLA_BFCL_REPETITION_PENALTY": "1.0",
                },
                clear=False,
            ),
            mock.patch(
                "benchmarks.bfcl.handler._request_session",
                return_value=session,
            ),
        ):
            result, _ = handler._query_prompting(
                {"function": [], "message": []}
            )

        handler.client.completions.create.assert_not_called()
        self.assertEqual(result.choices[0].text, "generated answer")
        self.assertEqual(result.usage.prompt_tokens, 7)
        self.assertEqual(result.usage.completion_tokens, 3)
        session.post.assert_called_once_with(
            "http://router/generate",
            json={
                "model": "test/model",
                "text": "rendered prompt",
                "sampling_params": {
                    "max_new_tokens": 123,
                    "temperature": 0.001,
                    "repetition_penalty": 1.0,
                    "top_p": 0.9,
                    "skip_special_tokens": False,
                },
            },
            timeout=(30, 1800),
        )

    def test_native_generate_does_not_retry_client_errors(self) -> None:
        response = mock.Mock(
            ok=False,
            status_code=400,
            text="input is too long",
        )
        session = mock.Mock()
        session.post.return_value = response

        with mock.patch(
            "benchmarks.bfcl.handler._request_session",
            return_value=session,
        ):
            with self.assertRaisesRegex(RuntimeError, "input is too long"):
                _post_native_generate("http://router/generate", {})

        session.post.assert_called_once()

    def test_native_generate_does_not_retry_context_error_reported_as_500(
        self,
    ) -> None:
        response = mock.Mock(
            ok=False,
            status_code=500,
            text=(
                "worker failed: The input (32889 tokens) is longer than the "
                "model's context length (32768 tokens)."
            ),
        )
        session = mock.Mock()
        session.post.return_value = response

        with (
            mock.patch(
                "benchmarks.bfcl.handler._request_session",
                return_value=session,
            ),
            mock.patch("benchmarks.bfcl.handler.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "not retried"):
                _post_native_generate("http://router/generate", {})

        session.post.assert_called_once()
        sleep.assert_not_called()

    def test_native_generate_still_retries_transient_server_errors(self) -> None:
        failed = mock.Mock(
            ok=False,
            status_code=503,
            text="worker temporarily unavailable",
        )
        valid = mock.Mock(ok=True)
        valid.json.return_value = {"text": "generated answer"}
        session = mock.Mock()
        session.post.side_effect = [failed, valid]

        with (
            mock.patch(
                "benchmarks.bfcl.handler._request_session",
                return_value=session,
            ),
            mock.patch("benchmarks.bfcl.handler.time.sleep"),
        ):
            result = _post_native_generate("http://router/generate", {})

        self.assertEqual(result, {"text": "generated answer"})
        self.assertEqual(session.post.call_count, 2)

    def test_native_generate_retries_malformed_json_responses(self) -> None:
        malformed = mock.Mock(ok=True, text='{"text": "unterminated')
        malformed.json.side_effect = json.JSONDecodeError(
            "Unterminated string",
            malformed.text,
            9,
        )
        valid = mock.Mock(ok=True)
        valid.json.return_value = {"text": "generated answer"}
        session = mock.Mock()
        session.post.side_effect = [malformed, valid]

        with (
            mock.patch(
                "benchmarks.bfcl.handler._request_session",
                return_value=session,
            ),
            mock.patch("benchmarks.bfcl.handler.time.sleep"),
        ):
            result = _post_native_generate("http://router/generate", {})

        self.assertEqual(result, {"text": "generated answer"})
        self.assertEqual(session.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
