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
