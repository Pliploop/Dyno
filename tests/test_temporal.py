import numpy as np
import torch
import inspect

from dyno.evaluation.temporal import compute_mspf


def _sequence() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.6, 0.8],
        ],
        dtype=torch.float32,
    )


def test_compute_mspf_uses_paper_contrast_defaults():
    defaults = inspect.signature(compute_mspf).parameters

    assert defaults["window"].default == 4
    assert defaults["power"].default == 3.0
    assert defaults["sigma"].default == 10.0
    assert defaults["lam"].default == 1e-3


def test_compute_mspf_normalizes_time_and_values_by_default():
    curve = compute_mspf(_sequence(), n_points=17)

    assert curve.shape == (17,)
    assert curve.dtype == np.float32
    assert np.isclose(curve.min(), 0.0)
    assert np.isclose(curve.max(), 1.0)


def test_compute_mspf_normalization_can_be_disabled():
    curve = compute_mspf(_sequence(), normalize=False)

    assert curve.shape == (4,)
    assert np.isclose(curve[0], 0.0)
    assert not np.isclose(curve.max(), 1.0)


def test_compute_mspf_preserves_legacy_absolute_modes():
    raw = compute_mspf(_sequence(), absolute=True, n_points=11)
    interpolated = compute_mspf(_sequence(), absolute=False, n_points=11)

    assert raw.shape == (4,)
    assert interpolated.shape == (11,)
    assert np.isclose(raw[0], 0.0)
    assert np.isclose(interpolated[0], 0.0)
    assert np.isclose(raw[-1], interpolated[-1])


def test_compute_mspf_maps_constant_curve_to_zero():
    sequence = torch.ones(5, 3)

    curve = compute_mspf(sequence, n_points=9)

    np.testing.assert_array_equal(curve, np.zeros(9, dtype=np.float32))
