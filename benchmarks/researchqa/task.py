from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "researchqa"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / DATA_FILE):
        sample_id = str(row["id"])
        rubrics = row["rubric"]
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError(f"ResearchQA sample {sample_id} has no rubric")
        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                data={
                    "query": str(row["query"]),
                    "date": str(row["date"]),
                    "rubric": rubrics,
                },
                meta={
                    "general_domain": str(row["general_domain"]),
                    "subdomain": str(row["subdomain"]),
                    "field": str(row["field"]),
                    "date": str(row["date"]),
                },
            )
        )
    return samples


# Official leaderboard submission instructions (linked from the ResearchQA README):
# prompt for "a system without default attribution behavior", i.e. a plain LLM.
OFFICIAL_PROMPT = """Answer the question completely and precisely in around 240-260 words.

## Citation Instructions
- Support statements with relevant papers and in-line citations
- A statement may need to be supported by multiple references and should then be cited as [1][2] (for example, "Paris is the capital of France [1][2]" where "1" and "2" are the first and second paper).

## Output Format
- Don't enumerate the facts. You should provide an answer in one to three paragraphs.
- All bibliography citations should list the paper title and year, separated by commas.
- Separate the answer and citations with two newlines ("\\n")

=== BEGIN EXAMPLE ===
[Your answer with in-line citations]

[1] Title of the paper (Year)
[2] Title of the paper (Year)
[3] Title of the paper (Year)
=== END EXAMPLE ===

Question: {question}
Answer:"""


def build_prompt(sample: Sample) -> str:
    return OFFICIAL_PROMPT.replace("{question}", str(sample.data["query"]))
