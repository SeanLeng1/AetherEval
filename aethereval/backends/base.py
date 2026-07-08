from typing import Any, Protocol

from aethereval.core.types import GenerationInput, GenerationOutput


class GenerationBackend(Protocol):
    name: str

    def generate(
        self,
        inputs: list[GenerationInput],
        gen_cfg: dict[str, Any],
    ) -> list[GenerationOutput]: ...

    def close(self) -> None: ...
