"""Shared tqdm bars for preprocessing, generation and scoring."""

from tqdm.auto import tqdm


class Progress(tqdm):
    def __init__(self, total, desc, unit="gen", enabled=True):
        super().__init__(
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not enabled,
        )
