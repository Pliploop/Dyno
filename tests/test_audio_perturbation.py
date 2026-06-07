import numpy as np

from dyno.evaluation.audio_perturbation import transform_audio


def test_audio_perturbations_are_deterministic_and_finite():
    sample_rate = 100
    audio = np.linspace(-1.0, 1.0, 4000, dtype=np.float32)
    for condition in (
        "gain",
        "pitch_shift",
        "time_stretch",
        "chunk_shuffle",
        "reverse",
        "section_delete",
    ):
        first = transform_audio(
            audio,
            sample_rate,
            condition,
            seed=4,
            chunk_seconds=2.0,
        )
        second = transform_audio(
            audio,
            sample_rate,
            condition,
            seed=4,
            chunk_seconds=2.0,
        )
        assert first.ndim == 1
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)


def test_gain_preserves_length_and_section_delete_shortens():
    audio = np.ones(1000, dtype=np.float32)

    gain = transform_audio(audio, 100, "gain", seed=0)
    deleted = transform_audio(audio, 100, "section_delete", seed=0, chunk_seconds=2.0)

    assert len(gain) == len(audio)
    assert len(deleted) < len(audio)
