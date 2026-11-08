"""Protocols between the pieces of the core. Implementations stay independent of each other."""
from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from jetson_vision.types import Detection, Frame, FrameResult, HealthState


@runtime_checkable
class FrameSource(Protocol):
    """live=True: frames arrive on their own clock and old ones may be dropped."""

    source_id: str
    live: bool

    def open(self) -> None: ...

    def read(self) -> Frame | None:
        """Next frame, or None at the end of a finite source. Raises SourceError if the device fails."""

    def close(self) -> None: ...


@runtime_checkable
class InferenceBackend(Protocol):
    model_id: str

    def load(self) -> None:
        """Refuses to load if the engine does not match the TensorRT version or the GPU."""

    def infer(self, input_u8_chw_or_hwc: np.ndarray) -> Mapping[str, np.ndarray]:
        """Takes the resized uint8 image, normalises on the GPU and returns raw outputs on the CPU."""

    def close(self) -> None: ...


@runtime_checkable
class Decoder(Protocol):
    def __call__(
        self,
        outputs: Mapping[str, np.ndarray],
        frame_width: int,
        frame_height: int,
        score_threshold: float,
    ) -> list[Detection]: ...


@runtime_checkable
class Sink(Protocol):
    def write(self, result: FrameResult) -> None: ...

    def write_health(self, state: HealthState) -> None: ...

    def close(self) -> None: ...


class SourceError(RuntimeError):
    """The source failed. The pipeline reconnects with backoff; it never ends the process."""
