import json
import os
import unittest
from unittest import mock

from benchmarks.bfcl._compat import ensure_bfcl_importable

ensure_bfcl_importable()

from benchmarks.bfcl.handlers import (  # noqa: E402
    OfficialPromptHandlerAdapter,
    ToolRLHandler,
)
from benchmarks.bfcl.handlers.common import (  # noqa: E402
    post_native_generate,
)
from benchmarks.bfcl.handlers.toolrl import (  # noqa: E402
    _convert_to_format_tool,
    _render_with_model_chat_template,
    _tool_call_tags_are_special,
)


class _TemplateTokenizer:
    def __init__(self, *, reject_system: bool = False):
        self.reject_system = reject_system
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        if self.reject_system and messages[0]["role"] == "system":
            raise ValueError("System role not supported")
        body = "\n".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return f"<model-template>{body}<assistant>"


class BfclHandlerTests(unittest.TestCase):
    def test_official_adapter_preserves_prompt_and_uses_native_generate(self) -> None:
        class UpstreamPromptHandler:
            def _format_prompt(self, messages, function):
                return f"official:{messages[0]['content']}:{len(function)}"

        class AdaptedHandler(OfficialPromptHandlerAdapter, UpstreamPromptHandler):
            pass

        handler = AdaptedHandler()
        handler.model_path_or_id = "official/model"
        handler.temperature = 0.2
        handler.max_context_length = 8192
        handler.tokenizer = mock.Mock()
        handler.tokenizer.tokenize.return_value = [1, 2]

        response = mock.Mock(ok=True)
        response.json.return_value = {
            "text": "official answer",
            "meta_info": {"prompt_tokens": 2, "completion_tokens": 2},
        }
        session = mock.Mock()
        session.post.return_value = response

        with (
            mock.patch.dict(
                os.environ,
                {
                    "AETHEREVAL_BFCL_GENERATE_URL": "http://router/generate",
                    "AETHEREVAL_BFCL_MAX_TOKENS": "64",
                },
                clear=False,
            ),
            mock.patch(
                "benchmarks.bfcl.handlers.common._request_session",
                return_value=session,
            ),
        ):
            result, _ = handler._query_prompting(
                {
                    "message": [{"role": "user", "content": "hello"}],
                    "function": [{"name": "tool"}],
                }
            )

        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["text"], "official:hello:1")
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 64)
        self.assertEqual(result.choices[0].text, "official answer")

    def test_tool_schema_matches_public_toolrl_prompt(self) -> None:
        tool = {
            "name": "search",
            "description": "Search with structured filters.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "dict",
                            "properties": {
                                "enabled": {"type": "boolean"},
                            },
                            "required": ["enabled"],
                        },
                    },
                },
                "required": ["query"],
            },
        }
        original = json.loads(json.dumps(tool))

        rendered = _convert_to_format_tool(tool)

        self.assertEqual(tool, original)
        self.assertNotIn("Required parameters:", rendered)
        self.assertNotIn("Optional parameters:", rendered)
        self.assertIn('"type": "integer"', rendered)
        self.assertIn('"type": "boolean"', rendered)
        self.assertIn('"required": ["enabled"]', rendered)

    def test_prompt_matches_public_toolrl_example(self) -> None:
        handler = ToolRLHandler.__new__(ToolRLHandler)
        handler.tokenizer = _TemplateTokenizer()
        prompt = handler._format_prompt(
            [{"role": "user", "content": "Use the tools."}],
            [],
        )

        self.assertIn('"name": "Tool name"', prompt)
        self.assertIn('"Parameter name": "Parameter content"', prompt)
        self.assertNotIn("integer/float as JSON numbers", prompt)
        self.assertNotIn("<|im_start|>", prompt)
        self.assertEqual(
            [message["role"] for message in handler.tokenizer.calls[0]["messages"]],
            ["system", "user"],
        )

    def test_multi_turn_suffix_is_used_before_first_tool_feedback(self) -> None:
        handler = ToolRLHandler.__new__(ToolRLHandler)
        handler.tokenizer = _TemplateTokenizer()
        messages = [{"role": "user", "content": "Start the task."}]

        multi_prompt = handler._render_prompt(
            messages,
            [],
            is_multi_turn=True,
        )
        single_prompt = handler._render_prompt(
            messages,
            [],
            is_multi_turn=False,
        )

        self.assertIn("comprehensive plan", multi_prompt)
        self.assertNotIn("given task in this turn", multi_prompt)
        self.assertIn("given task in this turn", single_prompt)
        self.assertNotIn("comprehensive plan", single_prompt)

    def test_pre_query_processing_records_multi_turn_per_example(self) -> None:
        handler = ToolRLHandler.__new__(ToolRLHandler)
        test_entry = {
            "id": "multi_turn_base_0",
            "function": [],
            "question": [[{"role": "user", "content": "Start."}]],
        }

        inference_data = handler._pre_query_processing_prompting(test_entry)

        self.assertIs(inference_data["is_multi_turn"], True)

    def test_format_prompt_ignores_benchmark_system_instructions_like_toolrl(
        self,
    ) -> None:
        handler = ToolRLHandler.__new__(ToolRLHandler)
        handler.tokenizer = _TemplateTokenizer()
        prompt = handler._format_prompt(
            [
                {"role": "system", "content": "Persistent memory instructions."},
                {"role": "user", "content": "What do you remember?"},
            ],
            [],
        )

        self.assertNotIn("Persistent memory instructions.", prompt)
        self.assertIn("What do you remember?", prompt)

    def test_gemma_style_template_folds_unsupported_system_role(self) -> None:
        tokenizer = _TemplateTokenizer(reject_system=True)
        with mock.patch.dict(
            os.environ,
            {"AETHEREVAL_BFCL_ENABLE_THINKING": "false"},
            clear=False,
        ):
            prompt = _render_with_model_chat_template(
                tokenizer,
                "ToolRL system instructions",
                "Dialogue history",
            )

        self.assertEqual(len(tokenizer.calls), 2)
        self.assertEqual(tokenizer.calls[0]["messages"][0]["role"], "system")
        self.assertEqual(tokenizer.calls[1]["messages"][0]["role"], "user")
        folded_content = tokenizer.calls[1]["messages"][0]["content"]
        self.assertIn("ToolRL system instructions", folded_content)
        self.assertIn("Dialogue history", folded_content)
        self.assertEqual(
            tokenizer.calls[1]["kwargs"],
            {"enable_thinking": False},
        )
        self.assertIn("<model-template>", prompt)
        self.assertNotIn("<|im_start|>", prompt)

    def test_missing_chat_template_fails_instead_of_assuming_qwen(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "usable chat template"):
            _render_with_model_chat_template(
                object(),
                "ToolRL system instructions",
                "Dialogue history",
            )

    def test_special_tool_tags_control_output_token_stripping(self) -> None:
        qwen_tokenizer = mock.Mock()
        qwen_tokenizer.all_special_tokens = ["<tool_call>", "</tool_call>"]
        gemma_tokenizer = mock.Mock()
        gemma_tokenizer.all_special_tokens = ["<bos>", "<eos>"]
        gemma_tokenizer.added_tokens_decoder = {}

        self.assertTrue(_tool_call_tags_are_special(qwen_tokenizer))
        self.assertFalse(_tool_call_tags_are_special(gemma_tokenizer))

    def test_native_generate_preserves_prompt_and_sampling(self) -> None:
        handler = ToolRLHandler.__new__(ToolRLHandler)
        handler.model_path_or_id = "test/model"
        handler.temperature = 0.001
        handler.max_context_length = 32768
        handler.skip_special_tokens = False
        handler.tokenizer = mock.Mock()
        handler.tokenizer.tokenize.return_value = [1, 2]
        handler.tokenizer.all_special_tokens = ["<tool_call>", "</tool_call>"]
        handler._render_prompt = mock.Mock(return_value="rendered prompt")
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
                    "AETHEREVAL_BFCL_GENERATE_URL": "http://router/generate",
                    "AETHEREVAL_BFCL_MAX_TOKENS": "123",
                    "AETHEREVAL_BFCL_TOP_P": "0.9",
                    "AETHEREVAL_BFCL_TOP_K": "-1",
                    "AETHEREVAL_BFCL_REPETITION_PENALTY": "1.0",
                },
                clear=False,
            ),
            mock.patch(
                "benchmarks.bfcl.handlers.common._request_session",
                return_value=session,
            ),
        ):
            result, _ = handler._query_prompting(
                {"function": [], "message": [], "is_multi_turn": False}
            )

        handler._render_prompt.assert_called_once_with(
            [],
            [],
            is_multi_turn=False,
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
            "benchmarks.bfcl.handlers.common._request_session",
            return_value=session,
        ):
            with self.assertRaisesRegex(RuntimeError, "input is too long"):
                post_native_generate("http://router/generate", {})

        session.post.assert_called_once()

    def test_toolrl_execution_keeps_json_value_types_strict(self) -> None:
        handler = ToolRLHandler.__new__(ToolRLHandler)
        result = (
            "<think>Call the tool.</think>\n<tool_call>\n"
            '{"name":"typed_tool","parameters":'
            '{"count":"2","enabled":"true"}}\n</tool_call>'
        )

        calls = handler.decode_execute(result)

        self.assertEqual(
            calls,
            ["typed_tool(count='2', enabled='true')"],
        )

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
                "benchmarks.bfcl.handlers.common._request_session",
                return_value=session,
            ),
            mock.patch("benchmarks.bfcl.handlers.common.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "not retried"):
                post_native_generate("http://router/generate", {})

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
                "benchmarks.bfcl.handlers.common._request_session",
                return_value=session,
            ),
            mock.patch("benchmarks.bfcl.handlers.common.time.sleep"),
        ):
            result = post_native_generate("http://router/generate", {})

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
                "benchmarks.bfcl.handlers.common._request_session",
                return_value=session,
            ),
            mock.patch("benchmarks.bfcl.handlers.common.time.sleep"),
        ):
            result = post_native_generate("http://router/generate", {})

        self.assertEqual(result, {"text": "generated answer"})
        self.assertEqual(session.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
