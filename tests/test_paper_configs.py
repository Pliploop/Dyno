from pathlib import Path

from hydra import compose, initialize_config_dir


def _compose(experiment: str):
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="train", overrides=[f"experiment={experiment}"])


def test_paper_default_uses_centered_residuals():
    cfg = _compose("paper_muq_1hz")

    assert cfg.model.latent_dim == 32
    assert cfg.model.output_mode == "centered_residuals"
    assert cfg.model.content_token == "mean"
    assert cfg.model.condition_z_tau is True
    assert cfg.model.condition_zc is False


def test_velocity_remains_an_explicit_ablation():
    cfg = _compose("paper_muq_1hz_velocity_d32")

    assert cfg.model.latent_dim == 32
    assert cfg.model.output_mode == "velocity"
    assert cfg.model.content_token == "first"


def test_no_anchor_predicts_absolute_embeddings_from_temporal_token_only():
    cfg = _compose("paper_muq_1hz_no_anchor")

    assert cfg.model.output_mode == "embeddings"
    assert cfg.model.condition_z_tau is True
    assert cfg.model.condition_zc is False
    assert cfg.model.predictor.condition_z_tau is True
    assert cfg.model.predictor.condition_zc is False


def test_paper_autoencoder_and_vae_are_explicit():
    ae = _compose("paper_muq_1hz_ae")
    vae = _compose("paper_muq_1hz_vae")

    assert ae.model.bottleneck._target_.endswith("DynoAutoEncoder")
    assert ae.model.beta == 0.0
    assert vae.model.bottleneck._target_.endswith("DynoBetaVAE")
    assert vae.model.beta == 0.01


def test_ready_encoder_ablation_configs_use_matching_dimensions():
    mert = _compose("paper_mert_1hz")
    music2latent = _compose("paper_music2latent_1hz")

    assert mert.data.embedding_encoder == "mert"
    assert mert.model.embedding_dim == 1024
    assert music2latent.data.embedding_encoder == "music2latent"
    assert music2latent.model.embedding_dim == 64


def test_structure_probe_callback_is_opt_in_and_uses_full_track_attention():
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=["experiment=paper_muq_1hz", "callbacks=paper_structure_probe"],
        )

    probe = cfg.callbacks.structure_probe
    assert probe.frame_rate == "1hz"
    assert "/muq/1hz/" in probe.datasets.salami
    assert probe.model_dim == 128
    assert probe.num_heads == 4
    assert probe.ffn_dim == 256
    assert probe.epochs == 100
    assert probe.warmup_epochs == 5
    assert probe.learning_rate == 1.0e-4
    assert probe.weight_decay == 0.01
    assert list(probe.probe_inputs) == ["local", "content", "temporal", "content_temporal"]
    assert probe.run_on_train_end is True
    assert probe.run_on_test_end is True
    assert probe.run_once is True
    assert "run_in_subprocess" not in probe


def test_paper_flipflop_uses_512_samples():
    cfg = _compose("paper_muq_1hz")

    assert cfg.callbacks.flipflop.n_flips == 512
    assert cfg.callbacks.flipflop.every_n_epochs == 5


def test_paper_mspf_defaults_are_consistent():
    cfg = _compose("paper_muq_1hz")

    reconstruction = cfg.callbacks.trajectory_reconstruction
    geometry = cfg.callbacks.annotation_free_temporal
    assert reconstruction.window == 4
    assert reconstruction.power == 3.0
    assert reconstruction.sigma == 10.0
    assert reconstruction.lam == 1.0e-3
    assert geometry.mspf_window == 4
    assert geometry.mspf_power == 3.0
    assert geometry.mspf_sigma == 10.0
    assert geometry.mspf_lam == 1.0e-3
