from pathlib import Path

import numpy as np
import pandas as pd

from dyno.evaluation.structure_probe import (
    ProbeTrack,
    ProbeWindowDataset,
    _boundary_f1,
    _pairwise_f1,
    _position_encoding,
    normalize_harmonix_function,
    parse_frame_rate,
)


def _track() -> ProbeTrack:
    return ProbeTrack(
        dataset="harmonix",
        track_id="track",
        features=np.ones((80, 4), dtype=np.float32),
        boundary_times=np.asarray([0.0, 10.0, 20.0, 40.0], dtype=np.float32),
        function_times=np.asarray([0.0, 10.0, 20.0, 40.0], dtype=np.float32),
        function_labels=["intro", "verse", "chorus", "outro"],
        fold=0,
        content=np.ones(3, dtype=np.float32),
        temporal=np.ones(2, dtype=np.float32),
    )


def test_probe_window_variants_have_expected_dimensions():
    vocabulary = {"intro": 0, "verse": 1, "chorus": 2, "outro": 3, "unknown": 4}
    expected = {
        "local": 4,
        "local_content": 7,
        "local_temporal": 6,
        "local_content_temporal": 9,
        "content_position": 11,
        "temporal_position": 10,
    }
    for probe_input, dimension in expected.items():
        dataset = ProbeWindowDataset(
            [_track()],
            [0],
            vocabulary,
            probe_input,
            frame_rate=2.0,
            window_seconds=30.0,
            hop_seconds=30.0,
            position_dim=8,
        )
        item = dataset[0]
        assert item["x"].shape == (60, dimension)
        assert item["mask"].sum() == 60


def test_probe_windows_follow_native_training_rate():
    vocabulary = {"intro": 0, "verse": 1, "chorus": 2, "outro": 3, "unknown": 4}
    dataset = ProbeWindowDataset(
        [_track()],
        [0],
        vocabulary,
        "local_temporal",
        frame_rate=0.1,
        window_seconds=30.0,
        hop_seconds=30.0,
        position_dim=8,
    )

    assert dataset[0]["x"].shape == (3, 6)


def test_embedding_rate_labels_parse_to_hz():
    assert parse_frame_rate("1hz") == 1.0
    assert parse_frame_rate("0.1hz") == 0.1


def test_boundary_f1_can_reproduce_trimmed_diagnostic():
    reference = np.asarray([0.0, 10.0, 20.0, 30.0])
    estimated = np.asarray([10.2, 19.8])

    assert _boundary_f1(reference, estimated, window=0.5, trim=True) == 1.0


def test_pairwise_f1_uses_mir_eval_segment_protocol():
    reference = np.asarray([0, 0, 1, 1, 2, 2])

    assert _pairwise_f1(reference, reference.copy(), frame_rate=2.0) == 1.0


def test_harmonix_functions_map_to_seven_class_vocabulary():
    assert normalize_harmonix_function("pre-chorus") == "chorus"
    assert normalize_harmonix_function("guitar solo") == "inst"
    assert normalize_harmonix_function("fade out") == "silence"


def test_checked_in_structure_folds_are_group_disjoint():
    path = Path(__file__).parents[1] / "configs/evaluation/folds/structure_seed142.csv"
    folds = pd.read_csv(path)

    assert set(folds["fold"]) == set(range(8))
    assert folds.groupby("group")["fold"].nunique().max() == 1
    assert set(folds["dataset"]) == {"salami", "harmonix"}


def test_position_encoding_is_deterministic():
    first = _position_encoding(60, 32)
    second = _position_encoding(60, 32)

    assert first.shape == (60, 32)
    np.testing.assert_array_equal(first, second)
