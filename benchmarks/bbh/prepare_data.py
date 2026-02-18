from __future__ import annotations

import json
import re
from pathlib import Path


BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

BBH_DESCRIPTIONS = {
    "boolean_expressions": "Evaluate the result of a random Boolean expression.",
    "causal_judgement": "Answer questions about causal attribution.",
    "date_understanding": "Infer the date from context.",
    "disambiguation_qa": "Clarify the meaning of sentences with ambiguous pronouns.",
    "dyck_languages": "Correctly close a Dyck-n word.",
    "formal_fallacies": "Distinguish deductively valid arguments from formal fallacies.",
    "geometric_shapes": "Name geometric shapes from their SVG paths.",
    "hyperbaton": "Order adjectives correctly in English sentences.",
    "logical_deduction_five_objects": (
        "A logical deduction task which requires deducing the order of a sequence of objects."
    ),
    "logical_deduction_seven_objects": (
        "A logical deduction task which requires deducing the order of a sequence of objects."
    ),
    "logical_deduction_three_objects": (
        "A logical deduction task which requires deducing the order of a sequence of objects."
    ),
    "movie_recommendation": "Recommend movies similar to the given list of movies.",
    "multistep_arithmetic_two": "Solve multi-step arithmetic problems.",
    "navigate": (
        "Given a series of navigation instructions, determine whether one would end up back "
        "at the starting point."
    ),
    "object_counting": "Questions that involve enumerating objects and asking the model to count them.",
    "penguins_in_a_table": "Answer questions about a table of penguins and their attributes.",
    "reasoning_about_colored_objects": (
        "Answer extremely simple questions about the colors of objects on a surface."
    ),
    "ruin_names": "Select the humorous edit that ruins the input movie or musical artist name.",
    "salient_translation_error_detection": (
        "Detect the type of error in an English translation of a German source sentence."
    ),
    "snarks": "Determine which of two sentences is sarcastic.",
    "sports_understanding": (
        "Determine whether an artificially constructed sentence relating to sports is plausible or not."
    ),
    "temporal_sequences": "Answer questions about which times certain events could have occurred.",
    "tracking_shuffled_objects_five_objects": (
        "Determine final positions of objects after a sequence of swaps."
    ),
    "tracking_shuffled_objects_seven_objects": (
        "Determine final positions of objects after a sequence of swaps."
    ),
    "tracking_shuffled_objects_three_objects": (
        "Determine final positions of objects after a sequence of swaps."
    ),
    "web_of_lies": "Evaluate a random boolean function expressed as a word problem.",
    "word_sorting": "Sort a list of words.",
}

_ANSWER_RE = re.compile(r"(?i)(?:so\\s+)?the\\s+answer\\s+is\\s*(.*?)(?:\\.\\s*)?$")


def _extract_gold_answer(target: str) -> str:
    text = str(target).strip()
    match = _ANSWER_RE.search(text)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer
    return text


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets is required for prepare_data.py. Install with `pip install datasets`."
        ) from exc

    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for subset in BBH_TASKS:
        ds = load_dataset("lukaemon/bbh", subset, split="test")
        for idx, row in enumerate(ds):
            input_text = str(row["input"]).strip()
            target = str(row["target"]).strip()
            if not input_text or not target:
                raise ValueError(f"Invalid BBH row for subset={subset} idx={idx}")

            answer = _extract_gold_answer(target)
            rows.append(
                {
                    "id": f"{subset}_{idx:05d}",
                    "subset": subset,
                    "input": input_text,
                    "target": target,
                    "answer": answer,
                    "description": BBH_DESCRIPTIONS.get(subset, ""),
                    "source": "lukaemon/bbh",
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)} subsets={len(BBH_TASKS)}")


if __name__ == "__main__":
    main()
