"""
Dyno validation callbacks.

MSPFCallback  — plots Music Semantic Progress Function for original vs reconstructed
SSMCallback   — plots Self-Similarity Matrices for original vs reconstructed
"""

import random
import logging
import numpy as np
import torch

from lightning.pytorch.callbacks import Callback

from dyno.callbacks.utils import BaseCallback
from dyno.evaluation.temporal import compute_mspf, compute_ssm, linearity_score

log = logging.getLogger(__name__)


def _get_wandb_logger(trainer):
    try:
        from lightning.pytorch.loggers import WandbLogger
        for lg in (trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]):
            if isinstance(lg, WandbLogger):
                return lg
    except Exception:
        pass
    return None


def _fig_to_wandb_image(fig):
    import wandb
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return wandb.Image(buf)


def _crop_valid(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Return only the valid (non-padding) frames of x given a bool mask."""
    if mask is None:
        return x
    return x[mask]


class MSPFCallback(BaseCallback):
    """
    At each validation epoch (every ``every_n_epochs`` epochs) selects
    ``n_samples`` random sequences from the first validation batch, computes
    the MSPF for both the original and Dyno-reconstructed sequence, and logs:

    - A line plot of original vs reconstructed MSPF (+ ideal linear reference)
    - Scalar linearity scores for both (logged via pl_module.log)

    Parameters are forwarded to :func:`dyno.evaluation.temporal.compute_mspf`.
    """

    def __init__(
        self,
        n_samples: int = 4,
        every_n_epochs: int = 1,
        window: int = 30,
        sigma: float = 10.0,
        lam: float = 1e-3,
        power: float = 1.0,
        absolute: bool = True,
        n_points: int = 100,
    ):
        super().__init__(every_n_epochs=every_n_epochs)
        self.n_samples = n_samples
        self.window = window
        self.sigma = sigma
        self.lam = lam
        self.power = power
        self.absolute = absolute
        self.n_points = n_points
        self._samples: list[tuple] = []   # (x_cpu, x_hat_cpu, mask_cpu)
        self._collected = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self._samples = []
        self._collected = False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._check_epoch(trainer, pl_module):
            return
        if self._collected:
            return

        x = batch["audio"]                          # (B, T, D)
        mask = batch.get("attention_mask", None)    # (B, T) bool | None

        B = x.shape[0]
        n = min(self.n_samples, B)
        indices = random.sample(range(B), n)

        with torch.no_grad():
            _, _, _, x_hat = pl_module(x, mask=mask)

        for i in indices:
            m = mask[i].cpu() if mask is not None else None
            self._samples.append((
                x[i].cpu(),
                x_hat[i].cpu(),
                m,
            ))

        self._collected = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._check_epoch(trainer, pl_module) or not self._samples:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wandb_logger = _get_wandb_logger(trainer)
        lin_scores_orig, lin_scores_recon = [], []
        figures = []

        for idx, (x, x_hat, mask) in enumerate(self._samples):
            x_v    = _crop_valid(x,    mask)   # (T_valid, D)
            x_hat_v = _crop_valid(x_hat, mask)

            if x_v.shape[0] < 2:
                continue

            mspf_kw = dict(window=self.window, sigma=self.sigma, lam=self.lam,
                           power=self.power, absolute=self.absolute, n_points=self.n_points)
            S_orig  = compute_mspf(x_v,    **mspf_kw)
            S_recon = compute_mspf(x_hat_v, **mspf_kw)

            ls_orig  = linearity_score(S_orig)
            ls_recon = linearity_score(S_recon)
            lin_scores_orig.append(ls_orig)
            lin_scores_recon.append(ls_recon)

            T = len(S_orig)
            x_axis = np.linspace(0.0, 1.0, T) if not self.absolute else np.arange(T)
            ideal = np.linspace(0, max(float(S_orig[-1]), 1e-8), T)
            xlabel = "normalised time" if not self.absolute else "frame"
            mode_tag = "relative" if not self.absolute else "absolute"

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(x_axis, S_orig,  label=f"original (lin={ls_orig:.2f})",  color="#2196F3")
            ax.plot(x_axis, S_recon, label=f"recon    (lin={ls_recon:.2f})", color="#FF5722")
            ax.plot(x_axis, ideal,   label="ideal linear",   color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("cumulative semantic progress")
            ax.set_title(f"MSPF ({mode_tag}) — sample {idx + 1}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            figures.append(fig)

        # Scalar logs (mean over samples)
        if lin_scores_orig:
            pl_module.log("val/mspf_linearity_orig",  float(np.mean(lin_scores_orig)),  on_epoch=True, sync_dist=True)
            pl_module.log("val/mspf_linearity_recon", float(np.mean(lin_scores_recon)), on_epoch=True, sync_dist=True)

        # W&B image logs
        if wandb_logger is not None and figures:
            import wandb
            wandb_logger.experiment.log(
                {"val/mspf": [_fig_to_wandb_image(f) for f in figures]},
                step=trainer.global_step,
            )

        for f in figures:
            plt.close(f)
        self._samples = []


class SSMCallback(BaseCallback):
    """
    At each validation epoch selects ``n_samples`` random sequences,
    computes the cosine Self-Similarity Matrix for both the original and the
    Dyno-reconstructed sequence, and logs side-by-side heatmaps to W&B.
    """

    def __init__(self, n_samples: int = 4, every_n_epochs: int = 1):
        super().__init__(every_n_epochs=every_n_epochs)
        self.n_samples = n_samples
        self._samples: list[tuple] = []
        self._collected = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self._samples = []
        self._collected = False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._check_epoch(trainer, pl_module):
            return
        if self._collected:
            return

        x = batch["audio"]
        mask = batch.get("attention_mask", None)

        B = x.shape[0]
        n = min(self.n_samples, B)
        indices = random.sample(range(B), n)

        with torch.no_grad():
            _, _, _, x_hat = pl_module(x, mask=mask)

        for i in indices:
            m = mask[i].cpu() if mask is not None else None
            self._samples.append((x[i].cpu(), x_hat[i].cpu(), m))

        self._collected = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._check_epoch(trainer, pl_module) or not self._samples:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wandb_logger = _get_wandb_logger(trainer)
        figures = []

        for idx, (x, x_hat, mask) in enumerate(self._samples):
            x_v     = _crop_valid(x,     mask)
            x_hat_v = _crop_valid(x_hat, mask)

            if x_v.shape[0] < 2:
                continue

            ssm_orig  = compute_ssm(x_v)
            ssm_recon = compute_ssm(x_hat_v)

            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            kw = dict(vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            im0 = axes[0].imshow(ssm_orig,  **kw)
            im1 = axes[1].imshow(ssm_recon, **kw)
            axes[0].set_title("original SSM")
            axes[1].set_title("reconstructed SSM")
            for ax in axes:
                ax.set_xlabel("frame")
                ax.set_ylabel("frame")
            fig.colorbar(im0, ax=axes, shrink=0.7, label="cosine similarity")
            fig.suptitle(f"SSM — sample {idx + 1}")
            fig.tight_layout()
            figures.append(fig)

        if wandb_logger is not None and figures:
            import wandb
            wandb_logger.experiment.log(
                {"val/ssm": [_fig_to_wandb_image(f) for f in figures]},
                step=trainer.global_step,
            )

        for f in figures:
            plt.close(f)
        self._samples = []
