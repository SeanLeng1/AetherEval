"""BFCL V3 model handler for ToolRL/GDPO-style models.

Models trained with the ToolRL recipe emit
``<think>...</think>\\n<tool_call>...</tool_call>\\n<response>...</response>``. BFCL's
default handlers don't speak that format, so we subclass bfcl_eval's prompt-mode
``OSSHandler`` to (1) render the ToolRL system+dialogue prompt and (2) decode the
``<tool_call>`` block into BFCL's AST / executable forms.

Ported from ToolRL ``benchmarks/BFCL/rlla_qwen.py`` and adapted to BFCL V3's
prompt-mode handler API.
"""

import json
import os
import re
import threading
import time
from types import SimpleNamespace

import requests
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import func_doc_language_specific_pre_processing
from overrides import override

from .errors import is_context_length_error
from .scoring import normalize_multi_turn_tool_calls


_REQUEST_STATE = threading.local()
_NATIVE_GENERATE_TIMEOUT = (30, 1800)


def _lenient_decode() -> bool:
    return os.getenv("RLLA_BFCL_LENIENT_DECODE", "0") == "1"


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _single_generate_result(result):
    while isinstance(result, list):
        if len(result) != 1:
            raise ValueError(
                "BFCL native generation returned an unexpected batch size: "
                f"{len(result)}"
            )
        result = result[0]
    if not isinstance(result, dict):
        raise TypeError(
            "BFCL native generation returned an unsupported response type: "
            f"{type(result).__name__}"
        )
    return result


def _generate_text(result: dict) -> str:
    for key in ("text", "output_text", "generated_text"):
        if key in result:
            return str(result[key])
    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs:
        return _generate_text(_single_generate_result(outputs[0]))
    raise ValueError("BFCL native generation response contains no generated text.")


def _token_count(result: dict, keys: tuple[str, ...]) -> int | None:
    for source in (result, result.get("meta_info")):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _request_session() -> requests.Session:
    session = getattr(_REQUEST_STATE, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _REQUEST_STATE.session = session
    return session


def _post_native_generate(url: str, payload: dict) -> dict:
    for attempt in range(3):
        try:
            response = _request_session().post(
                url,
                json=payload,
                timeout=_NATIVE_GENERATE_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
        else:
            if response.ok:
                try:
                    response_body = response.json()
                except ValueError as exc:
                    if attempt == 2:
                        raise RuntimeError(
                            "BFCL native generation returned malformed JSON after "
                            f"3 attempts: {response.text[:500]!r}"
                        ) from exc
                    time.sleep(0.5 * (2**attempt))
                    continue
                else:
                    return _single_generate_result(response_body)
            if is_context_length_error(response.text):
                raise RuntimeError(
                    "BFCL native generation rejected an overlength prompt "
                    f"(not retried), HTTP {response.status_code}: {response.text}"
                )
            if response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(
                    "BFCL native generation failed with HTTP "
                    f"{response.status_code}: {response.text}"
                )
            if attempt == 2:
                raise RuntimeError(
                    "BFCL native generation failed with HTTP "
                    f"{response.status_code}: {response.text}"
                )
        time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")


def _iter_json_objects(text):
    """Yield brace-balanced ``{...}`` substrings (handles multi-line / pretty JSON)."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def _parse_line_calls(text):
    calls = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict) or "name" not in call:
            continue
        if str(call["name"]).strip().lower() == "none":
            continue
        calls.append(call)
    return calls


def _extract_calls(result, lenient: bool = False):
    """Tool-call extraction -> list of ``{"name":..., "parameters":...}``.

    Faithful (default): only line-delimited JSON inside ``<tool_call>...</tool_call>``,
    matching ToolRL's handler and the GDPO Table 1 methodology.
    ``lenient=True`` (RLLA_BFCL_LENIENT_DECODE=1) additionally recovers calls the
    untrained base emits without tags (``` fences / bare JSON) -> measures raw capability
    rather than the format-penalized paper number; off by default.
    """
    if not isinstance(result, str):
        return []
    if "<tool_call>" in result:
        region = result.split("<tool_call>")[-1].split("</tool_call>")[0]
        return _parse_line_calls(region)
    elif lenient:
        region = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
    else:
        return []

    calls = []
    for obj in _iter_json_objects(region):
        try:
            d = json.loads(obj)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or "name" not in d:
            continue
        if str(d["name"]).strip().lower() == "none":
            continue
        calls.append(d)
    return calls


_TOOL_CALL_EXAMPLE = (
    '{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", '
    '"... ...": "... ..."}}\n'
    '{"name": "... ...", "parameters": {"... ...": "... ...", '
    '"... ...": "... ..."}}'
)

# Public ToolRL BFCL prompt, kept verbatim for comparable evaluation.
SYS = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{json_string}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists).
"""

_SINGLE_TURN_SUFFIX = (
    "If there's no appropriate tools to apply or required parameters are missing, please "
    "directly inform me in your response without any tool call, or call the tool with the "
    "name as 'None'. Otherwise, you should use one or more necessary tool calls to complete "
    "the given task in this turn."
)
_MULTI_TURN_SUFFIX = (
    "Use the one or more necessary tool calls to complete the task. You could perform tool "
    "calls for multiple rounds so you can try and error. Please make a comprehensive plan "
    "about how to achieve the goal step by step, and begin to call the tool step by step. If "
    "no tools apply or required parameters are missing, please also directly state this in "
    "your response without tool calls."
)
_RETRY_SUFFIX = (
    "If you think you have completed the current task, or the task cannot be finished, please "
    "respond directly without additional tool calls. If you encounter an error during tool "
    "execution or the task remains unfinished, retry with the one or more necessary tool calls "
    "according to your thought and plan until completion. Based on the tool execution feedback, "
    "reflect on if understanding or selectioin of tool is wrong, what tool calling step is "
    "missing, and how to achieve the task goal from now on."
)

# SGLang reserves a few positions for request bookkeeping/special tokens and rejects
# input at its advertised maximum-input boundary. Keep enough room for one generated
# token plus that server-side margin so unsupported BFCL cases become deterministic
# zero-score context overflows instead of HTTP 400 inference failures.
_SERVER_CONTEXT_MARGIN = 8


def _convert_to_format_tool(tools, count=1):
    if isinstance(tools, dict):
        params = tools["parameters"].get("properties", {})
        return (
            f"{count}. Name: {tools['name']}\nDescription: {tools['description']}\n"
            f"Parameters: {json.dumps(params)}"
        )
    if isinstance(tools, list):
        return "\n".join(_convert_to_format_tool(t, i + 1) for i, t in enumerate(tools))
    return tools


class RLLAHandler(OSSHandler):
    """Prompt-mode handler for ToolRL/GDPO models (think/tool_call/response format)."""

    def __init__(self, model_name, temperature, dtype="bfloat16"):
        super().__init__(model_name, temperature, dtype)
        self.is_fc_model = False
        # `<tool_call>`/`</tool_call>` are Qwen added-vocab special tokens; the backend
        # strips them under the default skip_special_tokens=True, leaving bare JSON the
        # decoder can't find. Keep them (same as bfcl's FC handlers, e.g. minicpm_fc).
        self.skip_special_tokens = False
        max_context_length = _env_int("RLLA_BFCL_MAX_CONTEXT_LENGTH")
        if max_context_length is not None:
            if max_context_length <= 0:
                raise ValueError("RLLA_BFCL_MAX_CONTEXT_LENGTH must be positive.")
            self.max_context_length = max_context_length

    @override
    def _format_prompt(self, messages, function, turn_type="single_turn"):
        """Render a standalone prompt through ToolRL's original handler API."""
        return self._render_prompt(
            messages,
            function,
            is_multi_turn=turn_type == "multi_turn",
        )

    def _render_prompt(self, messages, function, *, is_multi_turn):
        """Render a prompt with example-level interaction metadata."""
        tools = _convert_to_format_tool(function)
        system_prompt = SYS.format(tools=tools, json_string=_TOOL_CALL_EXAMPLE)

        user_prompt = "**Dialogue Records History**\n"
        for idx, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            if role == "system":
                continue
            if role == "user":
                suffix = _MULTI_TURN_SUFFIX if is_multi_turn else _SINGLE_TURN_SUFFIX
                user_prompt += f"<user> {str(content).strip()}\n{suffix} </user>\n"
            elif role == "tool":
                name = message.get("name", "").strip()
                user_prompt += (
                    f"<obs> You have made the tool call {name}. Execution returns: "
                    f"{str(content).strip()} </obs>\n"
                )
                if idx == len(messages) - 1:
                    user_prompt += f"\n<user> {_RETRY_SUFFIX} </user>\n"
            elif role == "assistant":
                user_prompt += f"\n{str(content).strip()}\n"
        user_prompt = user_prompt.strip()

        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        test_category = test_entry["id"].rsplit("_", 1)[0]
        functions = func_doc_language_specific_pre_processing(
            test_entry["function"],
            test_category,
        )

        return {
            "message": [],
            "function": functions,
            "is_multi_turn": test_category.startswith("multi_turn_"),
        }

    @override
    def _query_prompting(self, inference_data: dict):
        # Clean sampling: the sglang/vllm server otherwise fills repetition_penalty / top_p
        # / top_k from the model's generation_config (Qwen ships rep_penalty=1.1). Since the
        # ToolRL system prompt *contains* the `<tool_call>` example, a >1 repetition penalty
        # suppresses re-emitting `<tool_call>`, pushing the model to ``` fences instead.
        # ToolRL evaluated with offline vLLM (no such penalty); mirror that here.
        function = inference_data["function"]
        message = inference_data["message"]
        formatted_prompt = self._render_prompt(
            message,
            function,
            is_multi_turn=inference_data.get(
                "is_multi_turn",
                any(item.get("role") == "tool" for item in message),
            ),
        )
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        max_tokens = _env_int("RLLA_BFCL_MAX_TOKENS", 4096)
        if max_tokens is None or max_tokens <= 0:
            raise ValueError("RLLA_BFCL_MAX_TOKENS must be positive.")

        max_context_length = _env_int(
            "RLLA_BFCL_MAX_CONTEXT_LENGTH",
            self.max_context_length,
        )
        if max_context_length is None or max_context_length <= 0:
            raise ValueError("BFCL max context length must be positive.")

        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
        available_tokens = (
            max_context_length - input_token_count - _SERVER_CONTEXT_MARGIN
        )
        if available_tokens <= 0:
            raise ValueError(
                "BFCL prompt exceeds max context length: "
                f"input_tokens={input_token_count}, "
                f"max_context_length={max_context_length}."
            )
        leftover_tokens_count = min(max_tokens, available_tokens)

        extra_body = {
            "repetition_penalty": _env_float("RLLA_BFCL_REPETITION_PENALTY", 1.0),
            "top_p": _env_float("RLLA_BFCL_TOP_P", 1.0),
            "top_k": _env_int("RLLA_BFCL_TOP_K", -1),
        }
        seed = _env_int("RLLA_BFCL_SEED")
        if seed is not None:
            extra_body["seed"] = seed
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        start_time = time.time()
        generate_url = os.getenv("RLLA_BFCL_GENERATE_URL")
        if generate_url:
            sampling_params = {
                "max_new_tokens": leftover_tokens_count,
                "temperature": self.temperature,
                **extra_body,
            }
            if sampling_params.get("top_k") == -1:
                sampling_params.pop("top_k")
            result = _post_native_generate(
                generate_url,
                {
                    "model": self.model_path_or_id,
                    "text": formatted_prompt,
                    "sampling_params": sampling_params,
                },
            )
            generated_text = _generate_text(result)
            prompt_tokens = _token_count(
                result,
                ("prompt_tokens", "input_tokens", "input_token_count"),
            )
            completion_tokens = _token_count(
                result,
                (
                    "completion_tokens",
                    "output_tokens",
                    "output_token_count",
                    "num_output_tokens",
                ),
            )
            api_response = SimpleNamespace(
                choices=[SimpleNamespace(text=generated_text)],
                usage=SimpleNamespace(
                    prompt_tokens=(
                        input_token_count if prompt_tokens is None else prompt_tokens
                    ),
                    completion_tokens=(
                        len(self.tokenizer.tokenize(generated_text))
                        if completion_tokens is None
                        else completion_tokens
                    ),
                ),
            )
        else:
            api_response = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
                extra_body=extra_body,
                timeout=_NATIVE_GENERATE_TIMEOUT[1],
            )
        return api_response, time.time() - start_time

    @override
    def decode_ast(self, result, language="Python"):
        decoded = []
        for call in _extract_calls(result, _lenient_decode()):
            if "parameters" not in call:
                continue
            decoded.append({call["name"]: call["parameters"]})
        return decoded

    @override
    def decode_execute(self, result):
        calls = []
        decoded_calls = normalize_multi_turn_tool_calls(
            _extract_calls(result, _lenient_decode())
        )
        for c in decoded_calls:
            args = c.get("parameters", {})
            arg_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            calls.append(f"{c['name']}({arg_str})")
        return calls
