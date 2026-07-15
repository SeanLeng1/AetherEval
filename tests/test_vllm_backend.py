import unittest
from types import SimpleNamespace

import aethereval.backends.vllm.backend as vllm_backend


class _Tokenizer:
    def apply_chat_template(
        self,
        messages,  # noqa: ANN001
        tokenize,  # noqa: ANN001
        add_generation_prompt,  # noqa: ANN001
        enable_thinking=None,  # noqa: ANN001
    ):
        del tokenize, add_generation_prompt
        return f"thinking={enable_thinking}:{messages[-1]['content']}"


class _VLLMModule:
    class SamplingParams:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs


class _LLM:
    def __init__(self) -> None:
        self.prompts = None
        self.sampling_params = None

    def generate(self, *, prompts, sampling_params, use_tqdm):  # noqa: ANN001
        del use_tqdm
        self.prompts = prompts
        self.sampling_params = sampling_params
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 2],
                outputs=[SimpleNamespace(text="answer", token_ids=[3])],
            )
        ]


class VLLMBackendTests(unittest.TestCase):
    def test_run_generation_supports_thinking_and_no_thinking(self) -> None:
        payloads = [
            {
                "idx": 0,
                "sample_id": "a",
                "prompt": [{"role": "user", "content": "hello"}],
                "num_generations": 1,
            }
        ]
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                llm = _LLM()
                vllm_backend._run_generation(
                    llm=llm,
                    tokenizer=_Tokenizer(),
                    vllm_module=_VLLMModule,
                    payloads=payloads,
                    gen_cfg={
                        "n": 1,
                        "max_new_tokens": 8,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "enable_thinking": enabled,
                    },
                    show_progress=False,
                )

                self.assertEqual(llm.prompts, [f"thinking={enabled}:hello"])
                self.assertNotIn(
                    "enable_thinking",
                    llm.sampling_params.kwargs,
                )


if __name__ == "__main__":
    unittest.main()
