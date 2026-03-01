
from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "bbh"
DATA_FILE = "data/eval.jsonl"


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
    "snarks": (
        "Determine which of two sentences is sarcastic."
    ),
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


_REQUIRED_KEYS = {
    "id",
    "subset",
    "input",
    "target",
    "answer",
}


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)
    if not rows:
        raise RuntimeError(
            "BBH data file is empty or missing. Run "
            "`python benchmarks/bbh/prepare_data.py` to generate `data/eval.jsonl`."
        )

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("BBH row must be a JSON object")

        missing = sorted(_REQUIRED_KEYS - set(row.keys()))
        if missing:
            raise ValueError(f"BBH row missing keys: {', '.join(missing)}")

        sample_id = str(row["id"]).strip()
        subset = str(row["subset"]).strip()
        input_text = str(row["input"]).strip()
        target = str(row["target"]).strip()
        answer = str(row["answer"]).strip()

        if not sample_id:
            raise ValueError("BBH sample id must be non-empty")
        if not subset:
            raise ValueError(f"BBH subset is empty for sample {sample_id}")
        if not input_text:
            raise ValueError(f"BBH input is empty for sample {sample_id}")
        if not target:
            raise ValueError(f"BBH target is empty for sample {sample_id}")
        if not answer:
            raise ValueError(f"BBH answer is empty for sample {sample_id}")

        description = str(row.get("description", BBH_DESCRIPTIONS.get(subset, ""))).strip()

        samples.append(
            Sample(
                id=sample_id,
                gold=answer,
                meta={
                    "subset": subset,
                    "source": str(row.get("source", "lukaemon/bbh")).strip(),
                },
                data={
                    "subset": subset,
                    "input": input_text,
                    "target": target,
                    "answer": answer,
                    "description": description,
                },
            )
        )

    return samples


def build_prompt(sample: Sample) -> str:
    description = str(sample.data.get("description", "")).strip()
    input_text = str(sample.data["input"]).strip()

    parts: list[str] = []
    if description:
        parts.append(description)
    parts.append(f"Question: {input_text}")
    parts.append("Answer: Let's think step by step.")
    return "\n\n".join(parts)
