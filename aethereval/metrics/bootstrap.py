
import math
import random
from typing import Sequence


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    pos = q * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    weight = pos - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def bootstrap_mean(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        return {
            "mean": 0.0,
            "stderr": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n": 0.0,
        }

    mean = math.fsum(data) / n
    if n_resamples <= 0 or n == 1:
        return {
            "mean": mean,
            "stderr": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "n": float(n),
        }

    rng = random.Random(seed)
    sample_means: list[float] = []
    for _ in range(int(n_resamples)):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        sample_means.append(total / n)

    sample_means.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    stderr = 0.0
    m = len(sample_means)
    if m > 1:
        mu = math.fsum(sample_means) / m
        variance = math.fsum((x - mu) ** 2 for x in sample_means) / (m - 1)
        stderr = math.sqrt(max(variance, 0.0))

    return {
        "mean": mean,
        "stderr": stderr,
        "ci_low": _percentile(sample_means, alpha),
        "ci_high": _percentile(sample_means, 1.0 - alpha),
        "n": float(n),
    }
