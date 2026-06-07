from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dyno.evaluation.structure_probe import (
    AttentionStructureProbe,
    FullTrackProbeDataset,
    ProbeTrack,
    _boundary_f1,
    _joint_loss,
    _pairwise_f1,
    _position_encoding,
    collate_full_tracks,
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


def test_full_track_dataset_preserves_native_sequence():
    vocabulary = {"intro": 0, "verse": 1, "chorus": 2, "outro": 3, "unknown": 4}
    dataset = FullTrackProbeDataset([_track()], [0], vocabulary, frame_rate=2.0)
    item = dataset[0]

    assert item["local"].shape == (80, 4)
    assert item["boundary"].shape == (80,)
    assert item["function"].shape == (80,)


def test_full_track_collation_only_pads_batch():
    vocabulary = {"intro": 0, "verse": 1, "chorus": 2, "outro": 3, "unknown": 4}
    short = _track()
    short.features = short.features[:30]
    items = FullTrackProbeDataset([_track(), short], [0, 1], vocabulary, frame_rate=1.0)
    batch = collate_full_tracks([items[0], items[1]])

    assert batch["local"].shape == (2, 80, 4)
    assert batch["mask"][0].sum() == 80
    assert batch["mask"][1].sum() == 30


def test_global_token_probe_cannot_read_local_sequence():
    probe = AttentionStructureProbe(
        probe_input="temporal",
        local_dim=4,
        content_dim=3,
        temporal_dim=2,
        n_functions=5,
        model_dim=16,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
    ).eval()
    local = torch.randn(1, 12, 4)
    content = torch.randn(1, 3)
    temporal = torch.randn(1, 2)
    mask = torch.ones(1, 12, dtype=torch.bool)

    first = probe(local, content, temporal, mask)
    second = probe(local + 100.0, content, temporal, mask)

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_attention_padding_cannot_change_valid_predictions():
    probe = AttentionStructureProbe(
        probe_input="local_temporal",
        local_dim=4,
        content_dim=3,
        temporal_dim=2,
        n_functions=5,
        model_dim=16,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
    ).eval()
    local = torch.randn(1, 8, 4)
    changed = local.clone()
    changed[:, 5:] = 1.0e6
    content = torch.randn(1, 3)
    temporal = torch.randn(1, 2)
    mask = torch.tensor([[True, True, True, True, True, False, False, False]])

    first = probe(local, content, temporal, mask)
    second = probe(changed, content, temporal, mask)

    torch.testing.assert_close(first[0][:, :5], second[0][:, :5])
    torch.testing.assert_close(first[1][:, :5], second[1][:, :5])
    assert torch.count_nonzero(first[0][:, 5:]) == 0
    assert torch.count_nonzero(first[1][:, 5:]) == 0


def test_probe_loss_ignores_padding():
    mask = torch.tensor([[True, True, False]])
    boundary_logits = torch.tensor([[0.2, -0.3, 1.0e6]])
    function_logits = torch.randn(1, 3, 4)
    boundary = torch.tensor([[1.0, 0.0, 1.0]])
    function = torch.tensor([[2, 1, 3]])
    first = _joint_loss(
        boundary_logits,
        function_logits,
        boundary,
        function,
        mask,
    )
    boundary_logits[:, 2] = -1.0e6
    function_logits[:, 2] = 1.0e6
    second = _joint_loss(
        boundary_logits,
        function_logits,
        boundary,
        function,
        mask,
    )

    torch.testing.assert_close(first, second)


def test_all_probe_variants_produce_full_track_predictions():
    local = torch.randn(2, 10, 4)
    content = torch.randn(2, 3)
    temporal = torch.randn(2, 2)
    mask = torch.ones(2, 10, dtype=torch.bool)
    for probe_input in (
        "local",
        "content",
        "temporal",
        "content_temporal",
        "local_content",
        "local_temporal",
        "local_content_temporal",
    ):
        probe = AttentionStructureProbe(
            probe_input=probe_input,
            local_dim=4,
            content_dim=3,
            temporal_dim=2,
            n_functions=5,
            model_dim=16,
            num_heads=4,
            ffn_dim=32,
            dropout=0.0,
        )
        boundary, function = probe(local, content, temporal, mask)

        assert boundary.shape == (2, 10)
        assert function.shape == (2, 10, 5)


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
