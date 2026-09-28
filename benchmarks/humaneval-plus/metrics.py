import ast
import re
from typing import Any

from aethereval.core.types import Sample
from benchmark_utils.evalplus import SCORING_PROTOCOL, aggregate_base_plus, score_base_plus
from benchmark_utils.evalplus_sanitize import sanitize

PRIMARY_METRIC = "pass@1"
# EvalPlus sets a 4 GiB RLIMIT_AS in a checker started from the scoring process;
# score from a lean spawned worker even at num_proc=1 so the budget is the same.
SCORE_IN_SUBPROCESS = True

# Match fences with any info string so a ```text or ```bash block is consumed whole;
# otherwise its closing fence would open the next block and misalign every later pair.
_CODE_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_PYTHON_FENCE_TAGS = {"", "python", "py", "python3"}


def _candidate_solution(sample: Sample, generation: str) -> tuple[str, bool]:
    prompt = str(sample.data["prompt"])
    entry_point = str(sample.data["entry_point"])
    joiner = "" if prompt.endswith("\n") else "\n"
    try:
        code, full_solution = _assemble_blocks(generation, prompt, joiner, entry_point)
    except (RecursionError, ValueError):
        # Code too deep for ast (a thousand-term expression), or a null byte on
        # Python < 3.12: use EvalPlus's own extraction on the whole response.
        code, full_solution = generation, False
    return prompt + joiner + "\n" + sanitize(code, entrypoint=entry_point), full_solution


def _assemble_blocks(
    generation: str, prompt: str, joiner: str, entry_point: str
) -> tuple[str, bool]:
    blocks = [
        body
        for tag, body in _CODE_BLOCK_RE.findall(generation)
        if tag.strip().lower() in _PYTHON_FENCE_TAGS
    ]

    # Assemble declarations across blocks before the official sanitizer resolves
    # dependencies. Top-level usage examples are not part of the implementation.
    # A completion prompt may end at the function signature without a docstring.
    nodes = list(ast.parse(prompt + joiner + "    pass\n").body)
    full_solution = False
    for text in blocks or [generation]:
        body_only = text.lstrip("\n")[:1] in (" ", "\t")
        code = prompt + joiner + text.strip("\n") if body_only else text
        try:
            parsed = ast.parse(code)
        except SyntaxError:
            # Retain EvalPlus's prose/unfenced-code extraction fallback.
            try:
                parsed = ast.parse(sanitize(code))
            except SyntaxError:
                continue
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                full_solution = not body_only
            if isinstance(
                node,
                (ast.FunctionDef, ast.ClassDef, ast.Import,
                 ast.ImportFrom, ast.Assign, ast.AnnAssign),
            ):
                nodes.append(node)

    # sanitize keeps the first definition; resolve drafts beforehand so the last
    # function/class definition wins, including a final body after an earlier draft.
    seen = set()
    kept = []
    for node in reversed(nodes):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in seen:
                continue
            seen.add(node.name)
        kept.append(node)
    return ast.unparse(ast.Module(body=list(reversed(kept)), type_ignores=[])), full_solution


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    data = sample.data
    for split in ("base", "plus"):
        if not data[f"{split}_input"]:
            raise ValueError(f"{sample.id}: {split}_input is empty")
    solution, is_full_solution = _candidate_solution(sample, generation)
    prompt = str(data["prompt"])
    reference = (
        prompt
        + ("" if prompt.endswith("\n") else "\n")
        + str(data["canonical_solution"])
    )
    parsed = score_base_plus("humaneval", sample, solution, reference)
    return {
        "score": float(parsed["plus_pass"]),
        "is_pass": parsed["plus_pass"],
        "parsed": parsed,
        "meta": {
            **parsed,
            "scoring_protocol": SCORING_PROTOCOL,
            "full_solution": is_full_solution,
        },
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    return aggregate_base_plus(
        sample_results,
        metric_options,
        parsed_flag_fn=lambda record: isinstance(record.parsed, dict) and bool(record.parsed),
    )
