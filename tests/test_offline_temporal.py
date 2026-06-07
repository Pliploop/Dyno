import numpy as np

from dyno.evaluation.offline_temporal import (
    cross_encoder_mspf_geometry,
    retrieval_artifact,
)


def test_cross_encoder_geometry_logs_each_encoder():
    rng = np.random.default_rng(4)
    temporal = rng.normal(size=(5, 3)).astype(np.float32)
    features = {
        "muq": [rng.normal(size=(8, 4)).astype(np.float32) for _ in range(5)],
        "mert": [rng.normal(size=(8, 6)).astype(np.float32) for _ in range(5)],
    }

    metrics, curves = cross_encoder_mspf_geometry(
        temporal,
        features,
        mspf_points=12,
        mspf_max_frames=8,
        max_pairs=10,
    )

    assert set(curves) == {"muq", "mert"}
    assert curves["muq"].shape == (5, 12)
    assert "paper.mspf_cross_encoder/test/muq/geometry_spearman" in metrics
    assert "paper.mspf_cross_encoder/test/mert/geometry_spearman" in metrics


def test_retrieval_artifact_records_all_representation_modes():
    rng = np.random.default_rng(5)
    representations = {
        "content": rng.normal(size=(6, 4)).astype(np.float32),
        "temporal": rng.normal(size=(6, 3)).astype(np.float32),
    }
    mspf = rng.normal(size=(6, 10)).astype(np.float32)

    rows = retrieval_artifact(
        [f"track-{index}" for index in range(6)],
        representations,
        mspf,
        top_k=2,
        n_queries=2,
    )

    assert len(rows) == 2 * 3 * 2
    assert {row["representation"] for row in rows} == {
        "content",
        "temporal",
        "combined",
    }
