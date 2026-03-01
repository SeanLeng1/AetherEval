
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from types import ModuleType


PromptType = str | list[dict[str, str]]


@dataclass(slots=True)
class Sample:
    """Minimal sample schema shared between framework and tasks."""

    id: str
    gold: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationInput:
    sample_id: str
    prompt: PromptType
    num_generations: int = 1


@dataclass(slots=True)
class GenerationOutput:
    sample_id: str
    prompt: PromptType
    generations: list[str]
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRecord:
    sample_id: str
    gen_idx: int
    prompt: PromptType
    generation: str
    score: float
    is_pass: bool
    parsed: Any = None
    gold: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskSpec:
    name: str
    task_dir: Path
    task_module_path: Path
    metrics_module_path: Path


@dataclass(slots=True)
class TaskBundle:
    spec: TaskSpec
    task_module: ModuleType
    metrics_module: ModuleType


class TaskModule(Protocol):
    TASK_NAME: str
    DATA_FILE: str
    DEFAULT_GEN: dict[str, Any]

    def load_samples(self, task_dir: Path) -> list[Sample]:
        ...

    def build_prompt(self, sample: Sample) -> PromptType:
        ...


class MetricsModule(Protocol):
    def score_generation(self, sample: Sample, generation: str) -> dict[str, Any]:
        ...

    def aggregate(
        self,
        sample_results: list[dict[str, Any]],
        metric_options: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        ...
