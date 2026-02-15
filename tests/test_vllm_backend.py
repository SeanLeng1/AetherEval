from __future__ import annotations

import unittest
import warnings

import aethereval.vllm_backend as vllm_backend


class _TokenizerWithTemplate:
    def __init__(self) -> None:
        self.calls = 0
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # noqa: ANN001
        self.calls += 1
        self.last_messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "templated_prompt"


class _TokenizerBrokenTemplate:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # noqa: ANN001
        del messages, tokenize, add_generation_prompt
        raise ValueError("no chat template configured")


class VLLMBackendPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        vllm_backend._CHAT_TEMPLATE_FALLBACK_WARNED = False

    def test_string_prompt_is_wrapped_to_chat_and_uses_template(self) -> None:
        tokenizer = _TokenizerWithTemplate()
        out = vllm_backend._prompt_to_text("hello", tokenizer)
        self.assertEqual(out, "templated_prompt")
        self.assertEqual(tokenizer.calls, 1)
        self.assertEqual(
            tokenizer.last_messages,
            [{"role": "user", "content": "hello"}],
        )

    def test_missing_chat_template_falls_back_with_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = vllm_backend._prompt_to_text(
                [{"role": "user", "content": "hello"}],
                tokenizer=object(),
            )
        self.assertEqual(out, "user: hello")
        self.assertEqual(len(caught), 1)
        self.assertIn("falling back", str(caught[0].message))

    def test_broken_chat_template_falls_back_with_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = vllm_backend._prompt_to_text(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenizer=_TokenizerBrokenTemplate(),
            )
        self.assertEqual(out, "system: x\nuser: y")
        self.assertEqual(len(caught), 1)
        self.assertIn("apply_chat_template failed", str(caught[0].message))

    def test_fallback_warning_only_emitted_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = vllm_backend._prompt_to_text([{"role": "user", "content": "a"}], tokenizer=object())
            second = vllm_backend._prompt_to_text([{"role": "user", "content": "b"}], tokenizer=object())
        self.assertEqual(first, "user: a")
        self.assertEqual(second, "user: b")
        self.assertEqual(len(caught), 1)


if __name__ == "__main__":
    unittest.main()
