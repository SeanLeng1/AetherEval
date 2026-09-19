"""Repair RLLab/eval-set: AIME question text, and the two multiple-choice configs.

AIME: replace only the question text with MathArena transcriptions.

    python push.py          # dry run: verify answers, save previews, print changes
    python push.py --push   # upload both configs, attribution and repair notes

Retains the existing problem/solution schema, question order, gold strings and
math prompt suffix. Other configs are untouched. Does not rebuild AetherEval's
local data or run generation/scoring. Diagram code is removed; mathematical
formulas, tables and prose are retained. Previews are saved to `aime_rebuild/`.

gpqa-d: rebuilt from the official GPQA Diamond CSV (198 questions; the previous
config lacked one). Options follow simple-evals' permutation, `random.Random(0)`.
mmlu-pro-subset: rebuilt from a pinned TIGER-Lab/MMLU-Pro revision. It is the
official test split without the `math` category; gold letters are unchanged.

minervamath / olympiadbench: five gold answers that math-verify cannot parse are
rewritten in notation only (`GOLD_REWRITES`); values and problems are unchanged.
"""

import argparse
import hashlib
import io
import random
import re
import urllib.request
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download

REPO = "RLLab/eval-set"
COLUMNS = ["problem", "solution"]
SUFFIX = "\n\nPlease think step by step, and put your final answer within \\boxed{}."
SOURCES = {
    "aime24": ("MathArena/aime_2024_I", "MathArena/aime_2024_II"),
    "aime25": ("MathArena/aime_2025",),
}
MCQ_HEADER = (
    "The following are multiple choice questions (with answers){about}. Think step by "
    "step and then put your final answer option within \\boxed{{}}. Only put the letter "
    "in the box. There is only one correct answer.\n\n"
)
LETTERS = "ABCDEFGHIJ"
GPQA_CSV = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
GPQA_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
MMLU_PRO = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_EXCLUDED_CATEGORY = "math"
# \\boxed{...} contents that math-verify cannot parse, so that every correct answer
# scored 0. Same value, parseable notation. The three olympiadbench strings are
# verbatim from the official `final_answer` field (trailing period, stray "t",
# "m_max =" prefix).
GOLD_REWRITES = {
    "minervamath": {r"-1./3": r"-\frac{1}{3}", r"-3./2": r"-\frac{3}{2}"},
    "olympiadbench": {
        r"(-\infty, 0) \cup\{1\}.": r"(-\infty, 0) \cup\{1\}",
        r"t(0,4]": r"(0,4]",
        r"m_{\max }=n^{2}-n-1": r"n^{2}-n-1",
    },
}
DIAGRAM_CODE = re.compile(
    r"\[asy\].*?\[/asy\]|<asy>.*?</asy>|"
    r"\\begin\{asy\}.*?\\end\{asy\}|"
    r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
    re.DOTALL,
)


def strip_diagram_code(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", DIAGRAM_CODE.sub("\n\n", text)).strip()


def read_parquet(repo: str, filename: str, revision: str) -> pd.DataFrame:
    return pd.read_parquet(hf_hub_download(
        repo, filename, repo_type="dataset", revision=revision,
    ))


def build(config: str, old: pd.DataFrame, revisions: dict[str, str]) -> pd.DataFrame:
    rows = []
    for source in SOURCES[config]:
        frame = read_parquet(source, "data/train-00000-of-00001.parquet", revisions[source])
        expected = 15 if config == "aime24" else 30
        if sorted(frame["problem_idx"].tolist()) != list(range(1, expected + 1)):
            raise ValueError(f"{source}: missing or duplicate problem indices")
        # Keep the existing H4-derived 2024 order: I/II, then 1,10,...,15,2,...,9.
        key = (lambda values: values.astype(str)) if config == "aime24" else None
        rows.extend(frame.sort_values("problem_idx", key=key).to_dict("records"))
    if list(old.columns) != COLUMNS or len(old) != 30 or len(rows) != 30:
        raise ValueError(f"{config}: expected 30 rows with columns {COLUMNS}")
    for index, (row, gold) in enumerate(zip(rows, old["solution"]), 1):
        if int(row["answer"]) != int(gold):
            raise ValueError(f"{config} row {index}: answer/order mismatch: {gold} != {row['answer']}")
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            raise ValueError(f"{config} row {index}: empty problem")
    new = pd.DataFrame({
        "problem": [strip_diagram_code(row["problem"]) + SUFFIX for row in rows],
        "solution": old["solution"].tolist(),  # Preserve leading zeroes, too.
    })
    if new["problem"].duplicated().any():
        raise ValueError(f"{config}: duplicate problems")
    return new


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


def rewrite_golds(config: str, old: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply GOLD_REWRITES to the last \\boxed{...} of each solution; idempotent."""
    rewrites = GOLD_REWRITES[config]
    applied = dict.fromkeys(rewrites, 0)
    solutions = []
    for solution in old["solution"]:
        spans = boxes(solution)
        if spans:
            start, end = spans[-1]
            content = solution[start:end]
            if content in rewrites:
                applied[content] += 1
                solution = solution[:start] + rewrites[content] + solution[end:]
        solutions.append(solution)
    if any(count > 1 for count in applied.values()):
        raise ValueError(f"{config}: a gold rewrite matched several rows: {applied}")
    new = old.copy()
    new["solution"] = solutions
    return new, [f"{before} -> {rewrites[before]}" for before, count in applied.items() if count]


def render_mcq(about: str, question: str, options: list[str]) -> str:
    listed = "".join(f"{LETTERS[i]}. {option}\n" for i, option in enumerate(options))
    return f"{MCQ_HEADER.format(about=about)}Question:\n{question}\n\nOptions:\n{listed}"


def build_gpqa() -> pd.DataFrame:
    payload = urllib.request.urlopen(GPQA_CSV).read()
    if hashlib.sha256(payload).hexdigest() != GPQA_SHA256:
        raise ValueError("official GPQA Diamond CSV changed; re-audit before pushing")
    official = pd.read_csv(io.BytesIO(payload))
    if len(official) != 198:
        raise ValueError(f"expected 198 GPQA Diamond questions, got {len(official)}")
    # openai/simple-evals gpqa_eval.py: one permutation per example, in CSV order.
    rng = random.Random(0)
    rows = []
    for row in official.to_dict("records"):
        choices = [
            str(row[column]).strip()
            for column in (
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            )
        ]
        # Two official questions repeat one incorrect option; that is kept as released.
        if not all(choices) or choices.count(choices[0]) != 1:
            raise ValueError(f"GPQA {row['Record ID']}: empty choice or ambiguous gold")
        permuted = [choices[i] for i in rng.sample(range(4), 4)]
        rows.append(
            {
                "problem": render_mcq("", str(row["Question"]).strip(), permuted),
                "solution": LETTERS[permuted.index(choices[0])],
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def build_mmlu_pro(revision: str) -> pd.DataFrame:
    official = read_parquet(MMLU_PRO, "data/test-00000-of-00001.parquet", revision)
    if len(official) != 12032:
        raise ValueError(f"expected 12032 MMLU-Pro test questions, got {len(official)}")
    rows = []
    for row in official.to_dict("records"):
        if row["category"] == MMLU_PRO_EXCLUDED_CATEGORY:
            continue
        options = list(row["options"])
        if "N/A" in options or LETTERS[row["answer_index"]] != row["answer"]:
            raise ValueError(f"MMLU-Pro {row['question_id']}: unexpected options/answer")
        rows.append(
            {
                "problem": render_mcq(f" about {row['category']}", row["question"], options),
                "solution": row["answer"],
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def squash(text: str) -> str:
    return " ".join(text.split())


def question(problem: str) -> str:
    return squash(problem.split("\nOptions:\n")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    out = Path("aime_rebuild")

    api = HfApi()
    previous = api.dataset_info(REPO).sha
    revisions = {name: api.dataset_info(name).sha for names in SOURCES.values() for name in names}
    card_path = hf_hub_download(REPO, "README.md", repo_type="dataset", revision=previous)
    original_card = Path(card_path).read_text(encoding="utf-8")
    card = DatasetCard(original_card)
    infos = {info["config_name"]: info for info in card.data["dataset_info"]}
    operations = []
    changed_configs = []
    report = [f"Previous {REPO} revision: {previous}"]

    for config in SOURCES:
        filename = f"{config}/train-00000-of-00001.parquet"
        old = read_parquet(REPO, filename, previous)
        new = build(config, old, revisions)
        changed = sum(a != b for a, b in zip(old["problem"], new["problem"]))
        report.append(f"{config}: 30 -> 30 questions; {changed} prompts changed; all gold strings and row order retained.")
        # from_dict preserves the existing HF string schema (not pandas large_string).
        dataset = Dataset.from_dict({column: new[column].tolist() for column in COLUMNS})
        target = out / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(target)
        if not new.equals(old):
            changed_configs.append(config)
            operations.append(CommitOperationAdd(path_in_repo=filename, path_or_fileobj=target))
            info = infos[config]
            info["splits"] = [{"name": "train", "num_bytes": dataset.data.nbytes, "num_examples": 30}]
            info["download_size"] = target.stat().st_size
            info["dataset_size"] = dataset.data.nbytes

    mmlu_revision = api.dataset_info(MMLU_PRO).sha
    for config, new in (("gpqa-d", build_gpqa()), ("mmlu-pro-subset", build_mmlu_pro(mmlu_revision))):
        filename = f"{config}/train-00000-of-00001.parquet"
        old = read_parquet(REPO, filename, previous)[COLUMNS]
        if len(old) == len(new):
            # Same questions in the same order (MMLU-Pro repeats some question texts,
            # so rows are compared by position, not by text).
            pairs = list(zip(old["problem"], old["solution"], new["problem"], new["solution"]))
            if any(question(a) != question(c) for a, _, c, _ in pairs):
                raise ValueError(f"{config}: question order differs from the hub")
            added = 0
        else:
            previous_rows = {question(p): (p, g) for p, g in zip(old["problem"], old["solution"])}
            if len(previous_rows) != len(old):
                raise ValueError(f"{config}: duplicate question text; cannot match rows")
            pairs = [
                (*previous_rows[question(p)], p, g)
                for p, g in zip(new["problem"], new["solution"])
                if question(p) in previous_rows
            ]
            added = len(new) - len(pairs)
        report.append(
            f"{config}: {len(old)} -> {len(new)} questions; {added} added; of the "
            f"{len(pairs)} retained questions, {sum(a == c for a, _, c, _ in pairs)} are "
            f"byte-identical, {sum(squash(a) == squash(c) for a, _, c, _ in pairs)} identical "
            f"up to whitespace, and {sum(b == d for _, b, _, d in pairs)} keep their gold letter."
        )
        dataset = Dataset.from_dict({column: new[column].tolist() for column in COLUMNS})
        target = out / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(target)
        if not new.equals(old):
            changed_configs.append(config)
            operations.append(CommitOperationAdd(path_in_repo=filename, path_or_fileobj=target))
            info = infos[config]
            info["splits"] = [{"name": "train", "num_bytes": dataset.data.nbytes, "num_examples": len(new)}]
            info["download_size"] = target.stat().st_size
            info["dataset_size"] = dataset.data.nbytes

    for config in GOLD_REWRITES:
        filename = f"{config}/train-00000-of-00001.parquet"
        old = read_parquet(REPO, filename, previous)[COLUMNS]
        new, changes = rewrite_golds(config, old)
        if not new["problem"].equals(old["problem"]) or len(new) != len(old):
            raise ValueError(f"{config}: only gold notation may change")
        report.append(
            f"{config}: {len(old)} -> {len(new)} questions; problems and order unchanged; "
            + (f"gold notation rewritten: {'; '.join(changes)}." if changes else "gold rewrites already applied.")
        )
        if changes:
            changed_configs.append(config)
            dataset = Dataset.from_dict({column: new[column].tolist() for column in COLUMNS})
            target = out / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(target)
            operations.append(CommitOperationAdd(path_in_repo=filename, path_or_fileobj=target))
            info = infos[config]
            info["splits"] = [{"name": "train", "num_bytes": dataset.data.nbytes, "num_examples": len(new)}]
            info["download_size"] = target.stat().st_size
            info["dataset_size"] = dataset.data.nbytes

    source_links = "\n".join(
        f"- [{name}](https://huggingface.co/datasets/{name}/tree/{revision})"
        for name, revision in revisions.items()
    )
    previous_mirrors = (
        "Previous AIME mirrors used by AetherEval:\n\n"
        "- `aime24`: [HuggingFaceH4/aime_2024]"
        "(https://huggingface.co/datasets/HuggingFaceH4/aime_2024).\n"
        "- `aime25`: [yentinglin/aime_2025]"
        "(https://huggingface.co/datasets/yentinglin/aime_2025).\n\n"
        "The pre-migration `eval-set` configs matched those AetherEval copies in "
        "all 30 questions per year, row order and answers, ignoring whitespace "
        "and the appended math instruction. The original `eval-set` import "
        "provenance was not recorded, so these are identified as matching "
        "previous mirrors, not a verified import history."
    )
    preprocessing = (
        "Preprocessing: remove Asymptote blocks (`[asy]`, `<asy>` and the LaTeX "
        "`asy` environment) and TikZ `tikzpicture` environments. Preserve all "
        "prose, mathematical formulas and tables outside those blocks. Collapse "
        "excess blank lines, trim outer whitespace, and append the step-by-step / "
        "boxed-answer instruction. Question order and gold formatting are unchanged."
    )
    attribution = (
        "## AIME sources\n\n"
        "The `aime24` and `aime25` transcriptions are from [MathArena](https://matharena.ai/) "
        "by Jasper Dekoninck et al. Their [CC BY-NC-SA 4.0 license]"
        "(https://creativecommons.org/licenses/by-nc-sa/4.0/) applies to these two configs. "
        "The other configs retain their respective upstream licenses.\n\n"
        f"Pinned sources:\n\n{source_links}\n\n"
        f"{previous_mirrors}\n\n"
        f"{preprocessing}\n\n"
        "Reference: [Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs]"
        "(https://arxiv.org/abs/2605.00674)."
    )
    repairs = (
        "## Previous data repairs\n\n"
        "The original commits contain the complete row-level changes:\n\n"
        "- **OlympiadBench (675 → 674 questions):** previous content matched "
        "[knoveleng/OlympiadBench](https://huggingface.co/datasets/knoveleng/OlympiadBench), "
        "an older mirror; the original import provenance was not recorded. Rebuilt from "
        "`Hothan/OlympiadBench`, config `OE_TO_maths_en_COMP`. Updated 15 relay "
        "question transcriptions from the official release, removed one question "
        "removed upstream, and followed upstream's degree-unit formatting for one "
        "gold answer. [Full commit](https://huggingface.co/datasets/RLLab/eval-set/commit/"
        "93bfada9b7e597586ab6cdf68d98ff88ae4a3f3a).\n"
        "- **MinervaMath (272 questions, unchanged):** source "
        "[knoveleng/Minerva-Math](https://huggingface.co/datasets/knoveleng/Minerva-Math/"
        "tree/93d86e1f779d84c7393587c2cdbf6c7511591f95). The pre-repair "
        "`RLLab/eval-set` / `minervamath` rows match this source in all 272 questions, "
        "solution texts and row order, apart from the appended math instruction. "
        "Corrected the notation of "
        "62 gold answers (58 scientific-notation and 4 symbolic rewrites) so "
        "`math-verify` preserves their intended values. Question text, order and "
        "mathematical answers are unchanged. "
        "[Full commit](https://huggingface.co/datasets/RLLab/eval-set/commit/"
        "c774e009046b9173716066989a392d0851ea9e4a).\n"
        "- **Unparseable gold answers (5 rows):** `math-verify` could not parse these "
        "golds, so every correct answer scored 0. Notation only; values unchanged. "
        + " ".join(
            f"`{config}`: " + ", ".join(f"`{a}` → `{b}`" for a, b in rewrites.items()) + "."
            for config, rewrites in GOLD_REWRITES.items()
        )
        + " The olympiadbench strings are verbatim from the official `final_answer` field."
    )
    mcq_sources = (
        "## Multiple-choice sources\n\n"
        "- **`gpqa-d` (198 questions):** official GPQA Diamond, from the "
        f"[simple-evals CSV]({GPQA_CSV}) (sha256 `{GPQA_SHA256}`). The previous config "
        "held 197 of these questions; its import provenance and option shuffle were not "
        "recorded. Options now follow the permutation of openai/simple-evals "
        "`gpqa_eval.py` (`random.Random(0)`, `rng.sample(range(4), 4)` per question in "
        "CSV order), so gold letters differ from the previous revision. Question and "
        "option text is stripped of outer whitespace.\n"
        f"- **`mmlu-pro-subset` (10,681 questions):** [{MMLU_PRO}]"
        f"(https://huggingface.co/datasets/{MMLU_PRO}/tree/{mmlu_revision}) test split "
        f"without the `{MMLU_PRO_EXCLUDED_CATEGORY}` category (1,351 questions), in "
        "official order with official text. Gold letters are unchanged.\n\n"
        "Both use the prompt `The following are multiple choice questions (with answers)"
        "[ about <category>]. Think step by step ... \\boxed{}`, then `Question:`, a blank "
        "line, and `Options:` lettered from A."
    )
    for name, body in (
        ("matharena-aime", attribution),
        ("mcq-sources", mcq_sources),
        ("eval-set-repairs", repairs),
    ):
        start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
        section = f"{start}\n{body}\n{end}"
        if start in card.text:
            card.text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _, section=section: section, card.text, flags=re.DOTALL)
        else:
            card.text = card.text.rstrip() + "\n\n" + section + "\n"
    rebuilt_card = str(card)
    (out / "README.md").write_text(rebuilt_card, encoding="utf-8")
    if rebuilt_card != original_card:
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=rebuilt_card.encode()))

    report += [
        "", "Sources:", source_links,
        "", preprocessing,
        "", previous_mirrors,
        "", repairs,
        "", mcq_sources,
        "", "README also documents the previous OlympiadBench/MinervaMath rebuilds.",
        "", "Prompt changes require new generations; rescoring old answers is not equivalent.",
    ]
    description = "\n".join(report)
    print(description)
    print(f"Preview saved to {out.resolve()}")
    if not args.push:
        print(f"Dry run only. Re-run with --push to update {REPO}.")
    elif not operations:
        print("Already up to date; nothing to push.")
    else:
        result = api.create_commit(
            repo_id=REPO, repo_type="dataset", operations=operations,
            parent_commit=previous,
            commit_message=(
                f"repair {', '.join(changed_configs)} (see description)"
                if changed_configs
                else "README: document data sources and repairs"
            ),
            commit_description=description,
        )
        print(f"Pushed: {result.commit_url}")


if __name__ == "__main__":
    main()
