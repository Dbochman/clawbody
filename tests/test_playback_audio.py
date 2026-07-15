import numpy as np

from reachy_mini_openclaw.playback_audio import prepare_reachy_stereo


def test_prepares_stereo_without_halving_frame_duration() -> None:
    mono_pcm = np.arange(2_400, dtype=np.int16).reshape(1, -1)

    stereo = prepare_reachy_stereo(mono_pcm, 24_000, 16_000, volume=1.0)

    assert stereo.shape == (1_600, 2)
    assert stereo.dtype == np.float32
    np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])


def test_applies_playback_volume() -> None:
    mono_pcm = np.array([[16_384, -16_384]], dtype=np.int16)

    stereo = prepare_reachy_stereo(mono_pcm, 16_000, 16_000, volume=0.5)

    np.testing.assert_allclose(stereo[:, 0], [0.25, -0.25])
