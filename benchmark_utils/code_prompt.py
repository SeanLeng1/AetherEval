"""Code-generation prompt shared by HumanEval+, MBPP+ and LiveCodeBench.

The wording matches AetherRL's training prompt (data/process_code.py code_prompt), so
code tasks are evaluated in the format the policy was trained on: one user turn,
no system message.
"""

FUNCTION_INTERFACE = (
    "Return the completed Python function(s), preserving the requested names and signatures."
)
STDIN_INTERFACE = (
    "Read input from stdin and write the answer to stdout; do not hard-code the examples."
)
CODE_PLACEHOLDER = "# YOUR CODE HERE"


def build_code_prompt(
    question: str,
    *,
    fn_name: str | None = None,
    stdin: bool = False,
    code_template: str = CODE_PLACEHOLDER,
) -> list[dict[str, str]]:
    if stdin:
        interface = STDIN_INTERFACE
    else:
        interface = FUNCTION_INTERFACE
        if fn_name:
            interface += f" The tested callable is `{fn_name}`."
    content = (
        f"### Question:\n{question.strip()}\n\n### Format:\n"
        f"Please think step by step, then write the complete solution. {interface}\n"
        f"Put the final solution in one Python code block:\n```python\n{code_template}\n```"
    )
    return [{"role": "user", "content": content}]
