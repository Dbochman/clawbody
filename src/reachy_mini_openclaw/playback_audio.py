"""Audio framing helpers for Reachy Mini speaker playback."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample


def prepare_reachy_stereo(
    audio_data: NDArray,
    input_sample_rate: int,
    output_sample_rate: int,
    *,
    volume: float = 0.5,
) -> NDArray[np.float32]:
    """Convert mono int16 PCM to Reachy's interleaved float32 stereo frames."""
    mono = np.asarray(audio_data).reshape(-1).astype(np.float32) / 32768.0
    mono *= volume

    if input_sample_rate != output_sample_rate:
        output_frames = int(len(mono) * output_sample_rate / input_sample_rate)
        mono = resample(mono, output_frames).astype(np.float32)

    return np.repeat(mono[:, np.newaxis], 2, axis=1)
