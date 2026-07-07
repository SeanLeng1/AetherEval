"""BFCL-v3 model handler for ToolRL/GDPO-style models.

Models trained with the ToolRL recipe emit
``<think>...</think>\\n<tool_call>...</tool_call>\\n<response>...</response>``. BFCL's
default handlers don't speak that format, so we subclass bfcl_eval's prompt-mode
``OSSHandler`` to (1) render the ToolRL system+dialogue prompt and (2) decode the
``<tool_call>`` block into BFCL's AST / executable forms.

Ported from ToolRL ``benchmarks/BFCL/rlla_qwen.py`` and adapted to the new bfcl_eval
handler API (``_format_prompt(messages, function)`` — no ``turn_type`` arg;
``decode_ast(result, language, has_tool_call_tag)``).
"""

import json
import os
import re
import time

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


def _lenient_decode() -> bool:
    return os.getenv("RLLA_BFCL_LENIENT_DECODE", "0") == "1"


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


def _extract_calls(result, lenient: bool = False):
    """Tool-call extraction -> list of ``{"name":..., "parameters":...}``.

    Faithful (default): only the ``<tool_call>...</tool_call>`` block, matching ToolRL's
    handler and the GDPO Table 1 methodology (format adherence is part of the score).
    ``lenient=True`` (RLLA_BFCL_LENIENT_DECODE=1) additionally recovers calls the
    untrained base emits without tags (``` fences / bare JSON) -> measures raw capability
    rather than the format-penalized paper number; off by default.
    """
    if not isinstance(result, str):
        return []
    if "<tool_call>" in result:
        region = result.split("<tool_call>")[-1].split("</tool_call>")[0]
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
    '{"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}'
)

# ToolRL training system prompt (verbatim).
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
    "reflect on if understanding or selection of tool is wrong, what tool calling step is "
    "missing, and how to achieve the task goal from now on."
)


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
        self.is_fc_model = False  # prompt mode (ToolRL text format, not native FC)
        # `<tool_call>`/`</tool_call>` are Qwen added-vocab special tokens; the backend
        # strips them under the default skip_special_tokens=True, leaving bare JSON the
        # decoder can't find. Keep them (same as bfcl's FC handlers, e.g. minicpm_fc).
        self.skip_special_tokens = False

    @override
    def _format_prompt(self, messages, function):
        # Multi-turn once tool feedback has entered the history; single-turn otherwise.
        is_multi_turn = any(m.get("role") == "tool" for m in messages)
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
    def _query_prompting(self, inference_data: dict):
        # Clean sampling: the sglang/vllm server otherwise fills repetition_penalty / top_p
        # / top_k from the model's generation_config (Qwen ships rep_penalty=1.1). Since the
        # ToolRL system prompt *contains* the `<tool_call>` example, a >1 repetition penalty
        # suppresses re-emitting `<tool_call>`, pushing the model to ``` fences instead.
        # ToolRL evaluated with offline vLLM (no such penalty); mirror that here.
        function = inference_data["function"]
        message = inference_data["message"]
        formatted_prompt = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
        if self.max_context_length < input_token_count + 2:
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(4096, self.max_context_length - input_token_count - 2)

        extra_body = {"repetition_penalty": 1.0, "top_p": 1.0, "top_k": -1}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        start_time = time.time()
        api_response = self.client.completions.create(
            model=self.model_path_or_id,
            temperature=self.temperature,
            prompt=formatted_prompt,
            max_tokens=leftover_tokens_count,
            extra_body=extra_body,
            timeout=72000,
        )
        return api_response, time.time() - start_time

    @override
    def decode_ast(self, result, language="Python"):
        return [
            {c["name"]: c.get("parameters", c.get("arguments", {}))}
            for c in _extract_calls(result, _lenient_decode())
        ]

    @override
    def decode_execute(self, result):
        calls = []
        for c in _extract_calls(result, _lenient_decode()):
            args = c.get("parameters", c.get("arguments", {}))
            arg_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            calls.append(f"{c['name']}({arg_str})")
        return calls
