import numpy as np

from dyno.evaluation.audio_perturbation import (
    transform_audio,
    transform_embedding_sequence,
)


def test_audio_perturbations_are_deterministic_and_finite():
    sample_rate = 100
    audio = np.linspace(-1.0, 1.0, 4000, dtype=np.float32)
    for condition in (
        "gain",
        "pitch_shift",
        "time_stretch",
    ):
        first = transform_audio(
            audio,
            sample_rate,
            condition,
            seed=4,
        )
        second = transform_audio(
            audio,
            sample_rate,
            condition,
            seed=4,
        )
        assert first.ndim == 1
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)


def test_gain_preserves_length():
    audio = np.ones(1000, dtype=np.float32)

    gain = transform_audio(audio, 100, "gain", seed=0)

    assert len(gain) == len(audio)


def test_latent_perturbations_are_deterministic():
    sequence = np.arange(80, dtype=np.float32).reshape(20, 4)
    for condition in ("chunk_shuffle", "reverse", "section_delete"):
        first = transform_embedding_sequence(sequence, condition, seed=4, chunk_frames=4)
        second = transform_embedding_sequence(sequence, condition, seed=4, chunk_frames=4)

        assert first.ndim == 2
        assert first.shape[1] == sequence.shape[1]
        np.testing.assert_array_equal(first, second)

    reversed_sequence = transform_embedding_sequence(sequence, "reverse", seed=4)
    deleted = transform_embedding_sequence(sequence, "section_delete", seed=4, chunk_frames=4)
    np.testing.assert_array_equal(reversed_sequence, sequence[::-1])
    assert len(deleted) < len(sequence)
