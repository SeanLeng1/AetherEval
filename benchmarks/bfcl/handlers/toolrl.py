"""BFCL V3 handler for ToolRL/GDPO-style output."""

import json
import os
import re

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import func_doc_language_specific_pre_processing
from overrides import override

from .common import _env_int, query_rendered_prompt


def _lenient_decode() -> bool:
    return os.getenv("AETHEREVAL_BFCL_LENIENT_DECODE", "0") == "1"


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}.")


def _chat_template_kwargs() -> dict[str, bool]:
    enable_thinking = _env_bool("AETHEREVAL_BFCL_ENABLE_THINKING")
    return {"enable_thinking": enable_thinking} if enable_thinking is not None else {}


def _render_with_model_chat_template(tokenizer, system_prompt: str, user_prompt: str):
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError(
            "BFCL requires a tokenizer with a usable chat template; the selected "
            "model tokenizer has no apply_chat_template method."
        )

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        **_chat_template_kwargs(),
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    system_error = None
    try:
        rendered = apply_chat_template(messages, **kwargs)
    except Exception as exc:
        system_error = exc
    else:
        if (
            isinstance(rendered, str)
            and system_prompt in rendered
            and user_prompt in rendered
        ):
            return rendered
        system_error = ValueError("chat template dropped system or user content")

    # Gemma-family templates commonly reject the system role. Preserve the complete
    # ToolRL instruction in one user message, then use the model's own turn markers.
    folded_user_prompt = f"{system_prompt}\n\n{user_prompt}"
    try:
        rendered = apply_chat_template(
            [{"role": "user", "content": folded_user_prompt}],
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "BFCL could not render the selected model's chat template, either "
            "with a system role or with the system content folded into the user "
            f"message. System-role error: {type(system_error).__name__}: "
            f"{system_error}. Folded-message error: {type(exc).__name__}: {exc}."
        ) from exc

    if not isinstance(rendered, str) or folded_user_prompt not in rendered:
        raise RuntimeError(
            "BFCL model chat template did not preserve the ToolRL prompt content."
        )
    return rendered


def _tool_call_tags_are_special(tokenizer) -> bool:
    tool_tags = {"<tool_call>", "</tool_call>"}
    special_tokens = getattr(tokenizer, "all_special_tokens", ())
    if isinstance(special_tokens, (list, tuple, set)) and tool_tags.intersection(
        special_tokens
    ):
        return True

    added_tokens = getattr(tokenizer, "added_tokens_decoder", {})
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not isinstance(added_tokens, dict) or not callable(convert_tokens_to_ids):
        return False
    for token in tool_tags:
        token_id = convert_tokens_to_ids(token)
        if getattr(added_tokens.get(token_id), "special", False):
            return True
    return False


def _iter_json_objects(text):
    """Yield brace-balanced JSON object substrings."""
    depth = 0
    start = None
    for index, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
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
    """Decode ToolRL line-delimited JSON calls from the output tags."""
    if not isinstance(result, str):
        return []
    if "<tool_call>" in result:
        region = result.split("<tool_call>")[-1].split("</tool_call>")[0]
        return _parse_line_calls(region)
    if lenient:
        region = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
    else:
        return []

    calls = []
    for obj in _iter_json_objects(region):
        try:
            call = json.loads(obj)
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict) or "name" not in call:
            continue
        if str(call["name"]).strip().lower() == "none":
            continue
        calls.append(call)
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


def _convert_to_format_tool(tools, count=1):
    if isinstance(tools, dict):
        params = tools["parameters"].get("properties", {})
        return (
            f"{count}. Name: {tools['name']}\nDescription: {tools['description']}\n"
            f"Parameters: {json.dumps(params)}"
        )
    if isinstance(tools, list):
        return "\n".join(
            _convert_to_format_tool(tool, index + 1) for index, tool in enumerate(tools)
        )
    return tools


class ToolRLHandler(OSSHandler):
    """Prompt-mode handler for ToolRL's think/tool_call/response protocol."""

    def __init__(self, model_name, temperature, dtype="bfloat16"):
        super().__init__(model_name, temperature, dtype)
        self.is_fc_model = False
        max_context_length = _env_int("AETHEREVAL_BFCL_MAX_CONTEXT_LENGTH")
        if max_context_length is not None:
            if max_context_length <= 0:
                raise ValueError("AETHEREVAL_BFCL_MAX_CONTEXT_LENGTH must be positive.")
            self.max_context_length = max_context_length

    @override
    def _format_prompt(self, messages, function, turn_type="single_turn"):
        return self._render_prompt(
            messages,
            function,
            is_multi_turn=turn_type == "multi_turn",
        )

    def _render_prompt(self, messages, function, *, is_multi_turn):
        tools = _convert_to_format_tool(function)
        system_prompt = SYS.format(tools=tools, json_string=_TOOL_CALL_EXAMPLE)

        user_prompt = "**Dialogue Records History**\n"
        for index, message in enumerate(messages):
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
                if index == len(messages) - 1:
                    user_prompt += f"\n<user> {_RETRY_SUFFIX} </user>\n"
            elif role == "assistant":
                user_prompt += f"\n{str(content).strip()}\n"

        return _render_with_model_chat_template(
            self.tokenizer,
            system_prompt,
            user_prompt.strip(),
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
        return query_rendered_prompt(
            self,
            formatted_prompt,
            skip_special_tokens=not _tool_call_tags_are_special(self.tokenizer),
        )

    @override
    def decode_ast(self, result, language="Python"):
        decoded = []
        for call in _extract_calls(result, _lenient_decode()):
            if "parameters" in call:
                decoded.append({call["name"]: call["parameters"]})
        return decoded

    @override
    def decode_execute(self, result):
        calls = []
        for call in _extract_calls(result, _lenient_decode()):
            args = call.get("parameters", {})
            arg_str = ", ".join(f"{key}={value!r}" for key, value in args.items())
            calls.append(f"{call['name']}({arg_str})")
        return calls
