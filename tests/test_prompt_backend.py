import unittest
import warnings

import aethereval.backends.prompt as prompt_backend


class _TokenizerWithTemplate:
    def __init__(self) -> None:
        self.calls = 0
        self.last_messages = None
        self.last_enable_thinking = None

    def apply_chat_template(
        self,
        messages,  # noqa: ANN001
        tokenize,  # noqa: ANN001
        add_generation_prompt,  # noqa: ANN001
        enable_thinking=None,  # noqa: ANN001
    ):
        self.calls += 1
        self.last_messages = messages
        self.last_enable_thinking = enable_thinking
        assert tokenize is False
        assert add_generation_prompt is True
        return "templated_prompt"


class _TokenizerBrokenTemplate:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # noqa: ANN001
        del messages, tokenize, add_generation_prompt
        raise ValueError("no chat template configured")


class BackendPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        prompt_backend._CHAT_TEMPLATE_FALLBACK_WARNED = False

    def test_string_prompt_is_wrapped_to_chat_and_uses_template(self) -> None:
        tokenizer = _TokenizerWithTemplate()
        out = prompt_backend._prompt_to_text("hello", tokenizer)
        self.assertEqual(out, "templated_prompt")
        self.assertEqual(tokenizer.calls, 1)
        self.assertEqual(
            tokenizer.last_messages,
            [{"role": "user", "content": "hello"}],
        )

    def test_missing_chat_template_falls_back_with_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = prompt_backend._prompt_to_text(
                [{"role": "user", "content": "hello"}],
                tokenizer=object(),
            )
        self.assertEqual(out, "user: hello")
        self.assertEqual(len(caught), 1)
        self.assertIn("falling back", str(caught[0].message))

    def test_explicit_thinking_mode_is_forwarded_to_chat_template(self) -> None:
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                tokenizer = _TokenizerWithTemplate()
                prompt_backend._prompt_to_text(
                    "hello",
                    tokenizer,
                    {"enable_thinking": enabled},
                )
                self.assertIs(tokenizer.last_enable_thinking, enabled)

    def test_thinking_mode_is_not_forwarded_when_unspecified(self) -> None:
        tokenizer = _TokenizerWithTemplate()
        prompt_backend._prompt_to_text("hello", tokenizer)
        self.assertIsNone(tokenizer.last_enable_thinking)

    def test_generation_config_rejects_non_boolean_thinking_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            prompt_backend.chat_template_kwargs_from_generation_config(
                {"enable_thinking": "false"}
            )

    def test_broken_chat_template_falls_back_with_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = prompt_backend._prompt_to_text(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenizer=_TokenizerBrokenTemplate(),
            )
        self.assertEqual(out, "system: x\nuser: y")
        self.assertEqual(len(caught), 1)
        self.assertIn("apply_chat_template failed", str(caught[0].message))

    def test_fallback_warning_only_emitted_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = prompt_backend._prompt_to_text(
                [{"role": "user", "content": "a"}], tokenizer=object()
            )
            second = prompt_backend._prompt_to_text(
                [{"role": "user", "content": "b"}], tokenizer=object()
            )
        self.assertEqual(first, "user: a")
        self.assertEqual(second, "user: b")
        self.assertEqual(len(caught), 1)

    def test_fallback_requires_role_and_content(self) -> None:
        with self.assertRaises(KeyError):
            prompt_backend._prompt_to_text([{"role": "user"}], tokenizer=object())


if __name__ == "__main__":
    unittest.main()
