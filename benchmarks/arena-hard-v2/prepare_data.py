#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path


HF_ROOT = "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main/data/arena-hard-v2.0"
BASELINE_URL = f"{HF_ROOT}/model_answer/o3-mini-2025-01-31.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default="/tmp/arena-hard-auto/data/arena-hard-v2.0/question.jsonl",
    )
    parser.add_argument("--baseline", default=BASELINE_URL)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument(
        "--cohort-cache",
        default="/tmp/arena-hard-v2-cohort",
        help="Temporary download cache for official answer/judgment cohort.",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    questions = [json.loads(line) for line in Path(args.questions).open(encoding="utf-8")]
    baseline_rows = _read_jsonl(args.baseline)
    baseline = {row["uid"]: row for row in baseline_rows}
    hard_uids = {row["uid"] for row in questions if row["category"] == "hard_prompt"}

    with (output / "eval.jsonl").open("w", encoding="utf-8") as dst:
        for question in questions:
            if question["category"] != "hard_prompt":
                continue
            base = baseline[question["uid"]]
            payload = dict(question)
            payload["baseline_answer"] = base["messages"][-1]["content"]["answer"]
            payload["baseline_metadata"] = base["metadata"]
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")

    _prepare_style_cohort(output / "style_cohort.jsonl", Path(args.cohort_cache), hard_uids)


def _prepare_style_cohort(output: Path, cache: Path, hard_uids: set[str]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    answer_tree = _load_json(
        "https://huggingface.co/api/datasets/lmarena-ai/arena-hard-auto/tree/main/data/arena-hard-v2.0/model_answer?recursive=false&expand=false"
    )
    judgment_tree = _load_json(
        "https://huggingface.co/api/datasets/lmarena-ai/arena-hard-auto/tree/main/data/arena-hard-v2.0/model_judgment/gpt-4.1?recursive=false&expand=false"
    )
    answer_names = {Path(item["path"]).name for item in answer_tree}
    judgment_names = {Path(item["path"]).name for item in judgment_tree}
    names = sorted((answer_names & judgment_names) - {f"{BASELINE_MODEL}.jsonl"})

    with output.open("w", encoding="utf-8") as dst:
        for name in names:
            answer_path = cache / f"answer-{name}"
            judgment_path = cache / f"judgment-{name}"
            _download(f"{HF_ROOT}/model_answer/{name}", answer_path)
            _download(f"{HF_ROOT}/model_judgment/gpt-4.1/{name}", judgment_path)
            answer_rows = {
                row["uid"]: row for row in _read_jsonl(str(answer_path)) if row["uid"] in hard_uids
            }
            for judgment in _read_jsonl(str(judgment_path)):
                uid = judgment["uid"]
                if uid not in hard_uids or uid not in answer_rows:
                    continue
                games = judgment.get("games", [])
                if len(games) != 2 or any(game is None or game.get("score") is None for game in games):
                    continue
                game0, game1 = games
                battle_scores = LABEL_TO_SCORE[game1["score"]] + [
                    1.0 - value for value in LABEL_TO_SCORE[game0["score"]]
                ]
                payload = {
                    "uid": uid,
                    # Match upstream show_result.py, which collapses provider/path
                    # prefixes before fitting the style-control Bradley-Terry model.
                    "model": judgment["model"].split("/")[-1],
                    "battle_scores": battle_scores,
                    "model_metadata": answer_rows[uid]["metadata"],
                }
                dst.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(source: str) -> list[dict]:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            return [json.loads(line) for line in response]
    with Path(source).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _load_json(url: str):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def _download(url: str, output: Path) -> None:
    if output.exists() and output.stat().st_size:
        return
    partial = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as dst:
        while chunk := response.read(1024 * 1024):
            dst.write(chunk)
    partial.replace(output)


BASELINE_MODEL = "o3-mini-2025-01-31"
LABEL_TO_SCORE = {
    "A>B": [1.0], "A>>B": [1.0] * 3, "A=B": [0.5], "A<<B": [0.0] * 3,
    "A<B": [0.0], "B>A": [0.0], "B>>A": [0.0] * 3, "B=A": [0.5],
    "B<<A": [1.0] * 3, "B<A": [1.0],
}


if __name__ == "__main__":
    main()
