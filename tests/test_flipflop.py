import numpy as np

from dyno.callbacks.flipflop import _summary_metrics


def test_flipflop_summary_preserves_original_metrics_and_adds_spearman():
    rows = [
        {
            "extremeness": 0.1,
            "content_cosine_distance": 0.01,
            "temporal_cosine_distance": 0.2,
            "content_l2_distance": 0.1,
            "temporal_l2_distance": 1.0,
        },
        {
            "extremeness": 0.5,
            "content_cosine_distance": 0.02,
            "temporal_cosine_distance": 0.5,
            "content_l2_distance": 0.2,
            "temporal_l2_distance": 2.0,
        },
        {
            "extremeness": 0.9,
            "content_cosine_distance": 0.03,
            "temporal_cosine_distance": 0.8,
            "content_l2_distance": 0.3,
            "temporal_l2_distance": 3.0,
        },
    ]

    metrics = _summary_metrics(rows)

    assert np.isclose(metrics["content_cosine_distance"], 0.02)
    assert np.isclose(metrics["temporal_cosine_distance"], 0.5)
    assert np.isclose(metrics["temporal_minus_content_cosine_distance"], 0.48)
    assert np.isclose(metrics["content_l2_distance"], 0.2)
    assert np.isclose(metrics["temporal_l2_distance"], 2.0)
    assert np.isclose(metrics["flip_extremeness"], 0.5)
    assert np.isclose(metrics["content_shuffle_severity_spearman"], 1.0)
    assert np.isclose(metrics["temporal_shuffle_severity_spearman"], 1.0)
