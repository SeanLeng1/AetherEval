import json
from collections import defaultdict

from aethereval.metrics.common import mean

PRIMARY_METRIC = "accuracy"
MC_CATEGORIES = ("chat", "summary", "hallu")


def score_generation(sample, generation):
    # Upstream clean_response/json.loads; no mining answers from parse failures.
    text = generation.strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    field = (
        "OptimalOption"
        if sample.meta["category"] in MC_CATEGORIES
        else "CandidateAnswer"
    )
    try:
        result = json.loads(text)
        prediction = result.get(field) if isinstance(result, dict) else None
    except json.JSONDecodeError:
        prediction = None
    passed = (
        prediction is not None
        and not isinstance(prediction, bool)
        and prediction == sample.gold
    )
    return {
        "score": float(passed),
        "is_pass": passed,
        "parsed": {"prediction": prediction},
        "meta": {"parsed": prediction is not None},
    }


def aggregate(sample_results, metric_options=None):
    del metric_options
    scores, parsed, categories = [], [], defaultdict(list)
    for item in sample_results:
        values = [float(record["score"]) for record in item["records"]]
        if not values:
            continue
        value = mean(values)
        scores.append(value)
        parsed.extend(
            float((record.get("parsed") or {}).get("prediction") is not None)
            for record in item["records"]
        )
        categories[item["meta"]["category"]].append(value)
    return {
        "accuracy": mean(scores),
        "parsed_rate": mean(parsed),
        **{
            f"category/{name}": mean(values)
            for name, values in sorted(categories.items())
        },
    }
