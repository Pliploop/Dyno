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
