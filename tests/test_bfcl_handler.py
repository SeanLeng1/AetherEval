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
            timeout=72000,
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


if __name__ == "__main__":
    unittest.main()
