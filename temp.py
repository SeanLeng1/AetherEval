"""Rebuild the `olympiadbench` and `minervamath` configs of RLLab/eval-set.

    python temp.py            # dry run: build, diff against the hub, print the commits
    python temp.py --push     # one commit per config, carrying the printed message

Each commit's title and description are generated from the actual diff against the
current hub revision, so the history on
https://huggingface.co/datasets/RLLab/eval-set/commits/main states exactly what changed.

olympiadbench
    Current rows are a copy of knoveleng/OlympiadBench, an older snapshot of the
    official benchmark in which 15 ARML relay problems still read
    "Let $T=T N Y W R$" (unsolvable without the relayed number) and which keeps one
    problem (official id 1965, prose answer) that upstream has since removed.
    Rebuilt directly from Hothan/OlympiadBench `OE_TO_maths_en_COMP`.

minervamath
    Gold answers are kept, but their notation is rewritten where math-verify misreads
    it: `\\boxed{4.5e33}` parses as 4.5*e*33 (Euler's number) and a trailing one-letter
    factor is dropped as a unit (`\\frac{1}{m} t` -> 1/m). Values are unchanged.
"""

import argparse
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "RLLab/eval-set"
COLUMNS = ["problem", "solution"]
SUFFIX = "\n\nPlease think step by step, and put your final answer within \\boxed{}."
OFFICIAL_REPO = "Hothan/OlympiadBench"
OFFICIAL_CONFIG = "OE_TO_maths_en_COMP"
OFFICIAL_FILE = f"OlympiadBench/{OFFICIAL_CONFIG}/{OFFICIAL_CONFIG}.parquet"
RELAY_PLACEHOLDER = "T=T N Y W R"

E_NOTATION = re.compile(r"\s*(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)\s*")
POWER_OF_TEN = re.compile(r"(-?\d+(?:\.\d+)?) \\times 10\^\{(-?\d+)\}")
OLYMPIADBENCH_TITLE = "olympiadbench: rebuild from official Hothan/OlympiadBench"
OLYMPIADBENCH_WHY = """\
Source: {source_repo} @ {source}, config {source_config}.
Previous {repo} revision: {previous}

The previous rows were a copy of knoveleng/OlympiadBench, an older snapshot:
- 15 ARML relay problems read "Let $T=T N Y W R$" (the number you will receive) and could not be solved. Upstream now states T; the final answers are unchanged.
- 1 problem (official id 1965, a prose answer that cannot be graded automatically) was removed upstream and is dropped here.
- 1 gold follows upstream, which moved the degree sign to its `unit` field.

Format is unchanged: `problem` = official question + the step-by-step / \\boxed{{}} instruction, `solution` = "\\boxed{{" + final_answer without "$" + "}}"."""

MINERVAMATH_TITLE = (
    "minervamath: rewrite gold notation that math-verify misreads (values unchanged)"
)
MINERVAMATH_WHY = """\
Previous {repo} revision: {previous}

Problems are untouched. Only the contents of \\boxed{{...}} in `solution` change, and only in notation:
- e-notation golds: math-verify parses 4.5e33 as 4.5*e*33 (Euler's number), so a correct answer written as 4.5 \\times 10^{{33}} was graded wrong. Every rewritten value is exactly equal to the original (checked with Decimal).
- symbolic golds whose trailing one-letter factor math-verify dropped as a unit (\\frac{{1}}{{m}} t -> 1/m), or that used numpy syntax."""

# Same value, written so that math-verify keeps every factor.
MINERVA_BOX_REWRITES = {
    r"np.arcsin(10/13)": r"\arcsin(10/13)",
    r"\frac{37}{4} m": r"\frac{37 m}{4}",
    r"\frac{1}{m} t": r"\frac{t}{m}",
    r"\frac{1}{\sqrt{2}}c": r"\frac{c}{\sqrt{2}}",
}


def boxes(text: str) -> list[tuple[int, int]]:
    """(start, end) of the contents of every \\boxed{...}, with nested braces."""
    spans = []
    for match in re.finditer(r"\\boxed\{", text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        spans.append((match.end(), end - 1))
    return spans


def rewrite_boxes(text: str, rewrite) -> str:
    for start, end in reversed(boxes(text)):
        text = text[:start] + rewrite(text[start:end]) + text[end:]
    return text


def current(config: str, revision: str) -> pd.DataFrame:
    path = hf_hub_download(
        REPO,
        f"{config}/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision=revision,
    )
    return pd.read_parquet(path)[COLUMNS]


def build_olympiadbench(source_revision: str) -> pd.DataFrame:
    official = pd.read_parquet(
        hf_hub_download(
            OFFICIAL_REPO, OFFICIAL_FILE, repo_type="dataset", revision=source_revision
        )
    )
    rows = []
    for row in official.itertuples():
        if isinstance(row.context, str) and row.context.strip():
            raise ValueError(f"id {row.id}: text-only subset should have no context")
        answers = list(row.final_answer)
        if len(answers) != 1:
            raise ValueError(f"id {row.id}: expected one final_answer, got {answers}")
        if RELAY_PLACEHOLDER in row.question:
            raise ValueError(f"id {row.id}: relay placeholder still present upstream")
        rows.append(
            {
                "problem": row.question + SUFFIX,
                "solution": "\\boxed{" + answers[0].replace("$", "") + "}",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def build_minervamath(old: pd.DataFrame) -> pd.DataFrame:
    applied = dict.fromkeys(MINERVA_BOX_REWRITES, 0)

    def rewrite(box: str) -> str:
        match = E_NOTATION.fullmatch(box)
        if match:
            mantissa, exponent = match.group(1), int(match.group(2))
            if Decimal(box.strip()) != Decimal(mantissa).scaleb(exponent):
                raise ValueError(f"e-notation rewrite would change the value: {box}")
            return f"{mantissa} \\times 10^{{{exponent}}}"
        if box in MINERVA_BOX_REWRITES:
            applied[box] += 1
            return MINERVA_BOX_REWRITES[box]
        return box

    frame = old.copy()
    frame["solution"] = [rewrite_boxes(text, rewrite) for text in old["solution"]]
    if set(applied.values()) != {1}:
        raise ValueError(f"each symbolic rewrite must apply exactly once: {applied}")
    left = [
        text[start:end]
        for text in frame["solution"]
        for start, end in boxes(text)
        if E_NOTATION.fullmatch(text[start:end])
    ]
    if left:
        raise ValueError(f"e-notation golds left: {left}")
    return frame


def _outside_boxes(text: str) -> str:
    return rewrite_boxes(text, lambda _: "")


def changed_boxes(old_solution: str, new_solution: str) -> list[tuple[str, str]]:
    old_boxes = [old_solution[s:e] for s, e in boxes(old_solution)]
    new_boxes = [new_solution[s:e] for s, e in boxes(new_solution)]
    if len(old_boxes) != len(new_boxes) or _outside_boxes(
        old_solution
    ) != _outside_boxes(new_solution):
        raise ValueError("a solution changed outside of its \\boxed{...} contents")
    return [(o, n) for o, n in zip(old_boxes, new_boxes) if o != n]


def diff(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    for name, frame in (("current", old), ("rebuilt", new)):
        if frame["problem"].duplicated().any():
            raise ValueError(f"{name} rows cannot be matched: duplicate problem text")
    old_gold = dict(zip(old["problem"], old["solution"]))
    new_gold = dict(zip(new["problem"], new["solution"]))
    shared = [p for p in new["problem"] if p in old_gold]
    gold_changes = []
    for problem in shared:
        if old_gold[problem] != new_gold[problem]:
            gold_changes += changed_boxes(old_gold[problem], new_gold[problem])
    for before, after in gold_changes:
        # Every numeric rewrite must denote exactly the same number.
        match = POWER_OF_TEN.fullmatch(after)
        if E_NOTATION.fullmatch(before) and (
            not match
            or Decimal(before.strip())
            != Decimal(match.group(1)).scaleb(int(match.group(2)))
        ):
            raise ValueError(f"value changed: {before} -> {after}")
    return {
        "unchanged": sum(old_gold[p] == new_gold[p] for p in shared),
        "rows_with_gold_change": sum(old_gold[p] != new_gold[p] for p in shared),
        "gold_changes": gold_changes,
        "added": [p for p in new["problem"] if p not in old_gold],
        "removed": [p for p in old["problem"] if p not in new_gold],
        "order_kept": shared == [p for p in old["problem"] if p in new_gold],
    }


def short(problem: str, width: int = 100) -> str:
    text = " ".join(problem.removesuffix(SUFFIX).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def describe(old: pd.DataFrame, new: pd.DataFrame, why: str) -> str:
    stats = diff(old, new)
    lines = [
        why,
        "",
        f"Diff against the previous revision ({len(old)} -> {len(new)} rows):",
        f"- rows unchanged: {stats['unchanged']}",
        f"- rows whose gold was rewritten (same problem): {stats['rows_with_gold_change']}",
        f"- problems added: {len(stats['added'])}",
        f"- problems removed: {len(stats['removed'])}",
        f"- order of the remaining rows kept: {'yes' if stats['order_kept'] else 'no'}",
    ]
    if stats["gold_changes"]:
        lines += ["", "Gold rewrites (contents of \\boxed{...}):"]
        lines += [f"- {before} -> {after}" for before, after in stats["gold_changes"]]
    for title, problems in (("Added", stats["added"]), ("Removed", stats["removed"])):
        if problems:
            lines += ["", f"{title} problems:"] + [f"- {short(p)}" for p in problems]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--out", default="eval_set_rebuild")
    args = parser.parse_args()

    api = HfApi()
    previous = api.dataset_info(REPO).sha
    source = api.dataset_info(OFFICIAL_REPO).sha
    context = {
        "repo": REPO,
        "previous": previous,
        "source_repo": OFFICIAL_REPO,
        "source": source,
        "source_config": OFFICIAL_CONFIG,
    }

    configs = ("olympiadbench", "minervamath")
    old = {config: current(config, previous) for config in configs}
    commits = {
        "olympiadbench": (
            build_olympiadbench(source),
            OLYMPIADBENCH_TITLE,
            OLYMPIADBENCH_WHY.format(**context),
        ),
        "minervamath": (
            build_minervamath(old["minervamath"]),
            MINERVAMATH_TITLE,
            MINERVAMATH_WHY.format(**context),
        ),
    }

    out = Path(args.out)
    messages = {}
    for config, (frame, title, why) in commits.items():
        description = describe(old[config], frame, why)
        messages[config] = (title, description)
        target = out / config / "train-00000-of-00001.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        print(
            f"\n{'=' * 88}\ncommit for config `{config}`  (parquet preview: {target})"
        )
        print(f"{'-' * 88}\n{title}\n\n{description}")

    if not args.push:
        print(f"\n{'=' * 88}\nDry run only. Re-run with --push to upload to {REPO}.")
        return

    from datasets import Dataset

    for config, (frame, _, _) in commits.items():
        title, description = messages[config]
        # from_dict keeps the hub schema (string); from_pandas would yield large_string.
        dataset = Dataset.from_dict(
            {column: frame[column].tolist() for column in COLUMNS}
        )
        info = dataset.push_to_hub(
            REPO,
            config_name=config,
            split="train",
            commit_message=title,
            commit_description=description,
        )
        print(f"pushed {config}: {info.commit_url}")


if __name__ == "__main__":
    main()
