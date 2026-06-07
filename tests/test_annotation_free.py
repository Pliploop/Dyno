import numpy as np

from dyno.callbacks.annotation_free import (
    _dtw_distance,
    _linear_cka,
    _standardized_euclidean_distance_matrix,
)


def test_linear_cka_has_expected_references():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 8), dtype=np.float32)
    shuffled = x[rng.permutation(len(x))]

    assert np.isclose(_linear_cka(x, x), 1.0)
    assert _linear_cka(x, shuffled) < 0.5


def test_standardized_euclidean_is_scale_invariant():
    x = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    scaled = x * np.array([100.0, 0.01], dtype=np.float32)

    np.testing.assert_allclose(
        _standardized_euclidean_distance_matrix(x),
        _standardized_euclidean_distance_matrix(scaled),
        rtol=1e-5,
        atol=1e-6,
    )


def test_dtw_distance_handles_temporal_warping():
    reference = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    warped = np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32)
    different = np.array([1.0, 0.5, 0.0], dtype=np.float32)

    assert _dtw_distance(reference, warped) < _dtw_distance(reference, different)
