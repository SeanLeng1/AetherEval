"""Prepare the official QAMPARI test split with five oracle proof passages."""

import argparse
import json
import random
import re
import string
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARCHIVE_URL = "https://www.cs.tau.ac.il/~ohadr/qampari.zip"
_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def _normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    return " ".join(_ARTICLES.sub(" ", text).split())


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        columns = {
            key: column
            for key, column in value.items()
            if isinstance(column, Sequence) and not isinstance(column, (str, bytes))
        }
        size = max((len(column) for column in columns.values()), default=0)
        return [
            {
                key: column[index]
                if isinstance(column, Sequence) and not isinstance(column, (str, bytes))
                else column
                for key, column in value.items()
            }
            for index in range(size)
        ]
    return [dict(record) for record in value]


def _aliases(answer: Mapping[str, Any]) -> list[str]:
    raw_aliases = answer.get("aliases") or []
    if isinstance(raw_aliases, (str, bytes)):
        raw_aliases = [raw_aliases]
    values = [answer.get("answer_text", ""), *raw_aliases]
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def _list_safe_alias(answer: Mapping[str, Any]) -> str:
    """Return an alias representable by QAMPARI's comma-separated protocol."""

    return next(
        (
            alias
            for alias in _aliases(answer)
            if "," not in alias
            and "\n" not in alias
            and "\r" not in alias
            and _normalize(alias)
        ),
        "",
    )


def _answer_groups(row: Mapping[str, Any]) -> list[list[str]]:
    groups = [_aliases(answer) for answer in _records(row.get("answer_list"))]
    return [group for group in groups if group]


def _proof_passages(row: Mapping[str, Any], limit: int = 5) -> list[str]:
    passages: list[str] = []
    supported_aliases: set[str] = set()
    for answer in _records(row.get("answer_list")):
        aliases = {_normalize(alias) for alias in _aliases(answer)}
        aliases.discard("")
        if not _list_safe_alias(answer) or not aliases or aliases & supported_aliases:
            continue
        proofs = _records(answer.get("proof"))
        proof = next(
            (
                text
                for item in proofs
                if (text := str(item.get("proof_text", "")).strip())
                and text not in passages
            ),
            "",
        )
        if not proof:
            continue
        passages.append(proof)
        supported_aliases.update(aliases)
        if len(passages) == limit:
            break
    return passages


def _find_test_file(root: Path) -> Path:
    matches = list(root.rglob("test_data.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one QAMPARI test_data.jsonl below {root}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data/eval.jsonl"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="aethereval-qampari-") as temp:
        temp_dir = Path(temp)
        archive = args.archive or temp_dir / "qampari.zip"
        if not archive.exists():
            urllib.request.urlretrieve(ARCHIVE_URL, archive)

        extract_dir = temp_dir / "extracted"
        with zipfile.ZipFile(archive) as source:
            source.extractall(extract_dir)

        output_rows: list[dict[str, Any]] = []
        with _find_test_file(extract_dir).open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                row = json.loads(line)
                answer_groups = _answer_groups(row)
                passages = _proof_passages(row)
                question = str(
                    row.get("question_text") or row.get("question") or ""
                ).strip()
                if not question or len(answer_groups) < 5 or len(passages) != 5:
                    continue
                sample_id = str(row.get("qid") or index)
                random.Random(f"{args.seed}:{sample_id}").shuffle(passages)
                output_rows.append(
                    {
                        "id": sample_id,
                        "question": question,
                        "passages": passages,
                        "answer_groups": answer_groups,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in output_rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output_rows)} QAMPARI Oracle-5 test examples to {args.output}")


if __name__ == "__main__":
    main()
