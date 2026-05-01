import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from dyno.models.utils.base import BaseModule
from .aggregator import DynoAggregator
from .bottleneck import DynoBetaVAE, DynoAutoEncoder
from .predictor import DynoVelocityPredictor


class Dyno(BaseModule):
    """
    Dyno: disentangled dynamics model for audio embedding sequences.

    Pipeline
    --------
    x (B,T,D) → aggregator → h (B,D) → bottleneck → z_tau (B,latent_dim)
              → predictor(z_tau, content, T) → x_hat (B,T,D)

    Content token
    -------------
    "first" : x[:,0,:]          required for output_mode="velocity"
    "mean"  : masked mean over valid frames (mask-aware)

    Output modes
    ------------
    "velocity"   T-1 velocity vectors integrated from first frame → T reconstructed frames
    "embeddings" T full embedding predictions

    Loss
    ----
    MSE computed only on valid (non-padding) positions when mask is supplied.
    total = recon + beta * kl
    """

    def __init__(
        self,
        aggregator: DynoAggregator,
        bottleneck: DynoBetaVAE | DynoAutoEncoder,
        predictor: DynoVelocityPredictor,
        beta: float = 1.0,
        content_token: str = "first",
        output_mode: str = "velocity",
        ckpt_path: str | None = None,
        freeze: bool = False,
    ):
        if output_mode == "velocity" and content_token != "first":
            raise ValueError(
                "output_mode='velocity' requires content_token='first'. "
                "Use output_mode='embeddings' with content_token='mean'."
            )
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.aggregator = aggregator
        self.bottleneck = bottleneck
        self.predictor = predictor
        self.beta = beta
        self.content_token_mode = content_token
        self.output_mode = output_mode

    def get_content_token(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.content_token_mode == "first":
            return x[:, 0, :]
        if mask is not None:
            # Masked mean: average only valid frames
            m = mask.float().unsqueeze(-1)          # (B, T, 1)
            return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return x.mean(dim=1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.aggregator(x, mask=mask)
        return self.bottleneck(h)

    def decode(self, z_tau: torch.Tensor, content: torch.Tensor, T: int) -> torch.Tensor:
        if self.output_mode == "velocity":
            velocities = self.predictor(z_tau, content, num_frames=T - 1)
            x_hat = torch.cat(
                [content.unsqueeze(1), content.unsqueeze(1) + velocities.cumsum(dim=1)],
                dim=1,
            )
        else:
            x_hat = self.predictor(z_tau, content, num_frames=T)
        return x_hat

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, D = x.shape
        content = self.get_content_token(x, mask=mask)
        mu, log_var, z_tau = self.encode(x, mask=mask)
        x_hat = self.decode(z_tau, content, T)
        return mu, log_var, z_tau, x_hat

    def compute_loss(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var, z_tau, x_hat = self(x, mask=mask)
        kl_loss = self.bottleneck.kl_loss(mu, log_var)
        if mask is not None:
            recon_loss = F.mse_loss(x_hat[mask], x[mask])
        else:
            recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss


class LightningDyno(Dyno, LightningModule):
    """Lightning wrapper: optimizer/scheduler wiring and train/val loops."""

    def __init__(
        self,
        aggregator: DynoAggregator,
        bottleneck: DynoBetaVAE | DynoAutoEncoder,
        predictor: DynoVelocityPredictor,
        beta: float = 1.0,
        content_token: str = "first",
        output_mode: str = "velocity",
        optimizer=None,
        scheduler=None,
        ckpt_path: str | None = None,
        freeze: bool = False,
    ):
        LightningModule.__init__(self)
        Dyno.__init__(
            self,
            aggregator=aggregator,
            bottleneck=bottleneck,
            predictor=predictor,
            beta=beta,
            content_token=content_token,
            output_mode=output_mode,
            ckpt_path=ckpt_path,
            freeze=freeze,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        x = batch["audio"]                                  # (B, T, D)
        mask = batch.get("attention_mask", None)            # (B, T) bool or None
        total, recon, kl = self.compute_loss(x, mask=mask)
        on_step = stage == "train"
        self.log(f"{stage}/loss",     total,          on_step=on_step, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/recon",    recon,          on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/kl",       kl,             on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/beta_kl",  self.beta * kl, on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True)
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")
