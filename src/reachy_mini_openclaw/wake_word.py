"""Local openWakeWord gate for the direct Realtime microphone path."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

WAKE_SAMPLE_RATE = 16_000
WAKE_FRAME_SAMPLES = 1_280  # openWakeWord's preferred 80 ms frame
WakeWordState = Literal["sleeping", "waiting", "engaged"]


class WakeWordError(RuntimeError):
    """Raised when an enabled wake-word gate cannot be initialized safely."""


@dataclass(frozen=True)
class WakeWordDecision:
    forward: bool
    activated: bool = False
    timed_out: bool = False


def _ensure_openwakeword_assets() -> None:
    """Fetch the shared ONNX feature models without the unused classifier zoo."""
    try:
        import openwakeword
        from openwakeword.utils import download_file
    except ModuleNotFoundError as exc:
        raise WakeWordError(
            "openwakeword is required when REACHY_WAKE_WORD_ENABLED=true; "
            "install ClawBody with .[wake_word]"
        ) from exc

    target = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
    target.mkdir(parents=True, exist_ok=True)
    try:
        for metadata in openwakeword.FEATURE_MODELS.values():
            url = metadata["download_url"].replace(".tflite", ".onnx")
            if not (target / url.rsplit("/", 1)[-1]).exists():
                download_file(url, os.fspath(target))
    except Exception as exc:
        raise WakeWordError(f"could not prepare openWakeWord feature models: {exc}") from exc


class WakeWordGate:
    """Keep room audio local until the configured openWakeWord model fires."""

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.5,
        initial_timeout_seconds: float = 10.0,
        followup_timeout_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
        model_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        if not 0 < threshold <= 1:
            raise WakeWordError("wake-word threshold must be within (0, 1]")
        if initial_timeout_seconds <= 0 or followup_timeout_seconds <= 0:
            raise WakeWordError("wake-word timeouts must be positive")
        self.model_path = model_path
        self.threshold = threshold
        self.initial_timeout_seconds = initial_timeout_seconds
        self.followup_timeout_seconds = followup_timeout_seconds
        self._clock = clock
        self._model_factory = model_factory
        self._model: Any = None
        self._samples = np.empty(0, dtype=np.int16)
        self._state: WakeWordState = "sleeping"
        self._deadline: float | None = None

    @property
    def state(self) -> WakeWordState:
        return self._state

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise WakeWordError(f"wake-word model not found: {self.model_path}")
        if self._model_factory is not None:
            self._model = self._model_factory(self.model_path)
            return
        _ensure_openwakeword_assets()
        try:
            from openwakeword.model import Model

            self._model = Model(
                wakeword_models=[os.fspath(self.model_path)],
                inference_framework="onnx",
            )
        except Exception as exc:
            raise WakeWordError(f"could not load wake-word model: {exc}") from exc

    def sleep(self) -> None:
        self._state = "sleeping"
        self._deadline = None
        self._samples = np.empty(0, dtype=np.int16)
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass

    def mark_speech_started(self) -> None:
        if self._state in {"waiting", "engaged"}:
            self._state = "engaged"
            self._deadline = None

    def mark_speech_stopped(self) -> None:
        if self._state == "engaged":
            # A response should start promptly. This guard only prevents a failed
            # Realtime turn from leaving the microphone gate open indefinitely.
            self._deadline = self._clock() + 120.0

    def mark_response_started(self) -> None:
        if self._state == "engaged":
            self._deadline = None

    def mark_response_done(self) -> None:
        if self._state == "engaged":
            self._deadline = self._clock() + self.followup_timeout_seconds

    def process(self, audio: NDArray[np.int16]) -> WakeWordDecision:
        """Process 16 kHz mono PCM and decide whether this frame may be forwarded."""
        now = self._clock()
        timed_out = False
        if self._deadline is not None and now >= self._deadline:
            timed_out = True
            self.sleep()

        if self._state != "sleeping":
            return WakeWordDecision(forward=True, timed_out=timed_out)

        self.load()
        samples = np.asarray(audio, dtype=np.int16).reshape(-1)
        if samples.size:
            self._samples = np.concatenate((self._samples, samples))

        while self._samples.size >= WAKE_FRAME_SAMPLES:
            frame = self._samples[:WAKE_FRAME_SAMPLES]
            self._samples = self._samples[WAKE_FRAME_SAMPLES:]
            scores = self._model.predict(frame)
            score = max(scores.values()) if scores else 0.0
            if score < self.threshold:
                continue
            self._state = "waiting"
            self._deadline = now + self.initial_timeout_seconds
            self._samples = np.empty(0, dtype=np.int16)
            try:
                self._model.reset()
            except Exception:
                pass
            # Drop the detection frame so "Hey Claude" does not become a user
            # turn. The next microphone frame begins the command window.
            return WakeWordDecision(forward=False, activated=True, timed_out=timed_out)

        return WakeWordDecision(forward=False, timed_out=timed_out)
