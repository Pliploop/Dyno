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
    "centered_residuals" T residual predictions around the mean content token

    Loss
    ----
    Reconstruction loss computed only on valid (non-padding) positions when mask is supplied.
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
        embedding_dim: int | None = None,
        model_dim: int | None = None,
        latent_dim: int | None = None,
        input_norm: str | None = None,
        input_norm_eps: float = 1e-5,
        recon_loss: str = "l1",
        huber_delta: float = 1.0,
        ckpt_path: str | None = None,
        freeze: bool = False,
    ):
        valid_output_modes = ("velocity", "embeddings", "centered_residuals")
        if output_mode not in valid_output_modes:
            raise ValueError(f"output_mode must be one of: {', '.join(valid_output_modes)}")
        if output_mode == "velocity" and content_token != "first":
            raise ValueError(
                "output_mode='velocity' requires content_token='first'. "
                "Use output_mode='embeddings' with content_token='mean'."
            )
        if output_mode == "centered_residuals" and content_token != "mean":
            raise ValueError(
                "output_mode='centered_residuals' requires content_token='mean' "
                "so residual targets are centered on the same content token."
            )
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.aggregator = aggregator
        self.bottleneck = bottleneck
        self.predictor = predictor
        self.beta = beta
        self.content_token_mode = content_token
        self.output_mode = output_mode
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.recon_loss = recon_loss.lower()
        if self.recon_loss not in ("l1", "l2", "mse", "huber", "smooth_l1"):
            raise ValueError("recon_loss must be one of: l1, l2, mse, huber, smooth_l1")
        self.huber_delta = huber_delta
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        self.input_norm_mode = (input_norm or "none").lower()
        input_dim = getattr(self.aggregator, "input_dim", None)
        if input_dim is None:
            raise ValueError("Dyno input_norm requires aggregator.input_dim to be defined")
        if self.input_norm_mode in ("none", "identity"):
            self.input_norm = nn.Identity()
        elif self.input_norm_mode in ("layernorm", "layer_norm"):
            self.input_norm = nn.LayerNorm(input_dim, eps=input_norm_eps)
        else:
            raise ValueError("input_norm must be one of: none, identity, layernorm")

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_norm(x)

    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.recon_loss == "l1":
            return F.l1_loss(x_hat, x)
        if self.recon_loss in ("huber", "smooth_l1"):
            return F.huber_loss(x_hat, x, delta=self.huber_delta)
        return F.mse_loss(x_hat, x)

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

    def predict_target(self, z_tau: torch.Tensor, content: torch.Tensor, T: int) -> torch.Tensor:
        if self.output_mode == "velocity":
            return self.predictor(z_tau, content, num_frames=T - 1)
        return self.predictor(z_tau, content, num_frames=T)

    def decode_prediction(self, prediction: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "velocity":
            return torch.cat(
                [content.unsqueeze(1), content.unsqueeze(1) + prediction.cumsum(dim=1)],
                dim=1,
            )
        if self.output_mode == "centered_residuals":
            return content.unsqueeze(1) + prediction
        return prediction

    def reconstruction_target(self, x: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "centered_residuals":
            return x - content.unsqueeze(1)
        return x

    def decode(self, z_tau: torch.Tensor, content: torch.Tensor, T: int) -> torch.Tensor:
        prediction = self.predict_target(z_tau, content, T)
        return self.decode_prediction(prediction, content)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        x = self.normalize_input(x)
        B, T, D = x.shape
        content = self.get_content_token(x, mask=mask)
        mu, log_var, z_tau = self.encode(x, mask=mask)
        x_hat = self.decode(z_tau, content, T)
        return mu, log_var, z_tau, x_hat

    def compute_loss(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.normalize_input(x)
        B, T, D = x.shape
        content = self.get_content_token(x, mask=mask)
        mu, log_var, z_tau = self.encode(x, mask=mask)
        prediction = self.predict_target(z_tau, content, T)
        target = self.reconstruction_target(x, content)
        kl_loss = self.bottleneck.kl_loss(mu, log_var)
        if self.output_mode == "velocity":
            prediction = self.decode_prediction(prediction, content)
        if mask is not None:
            recon_loss = self.reconstruction_loss(prediction[mask], target[mask])
        else:
            recon_loss = self.reconstruction_loss(prediction, target)
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
        embedding_dim: int | None = None,
        model_dim: int | None = None,
        latent_dim: int | None = None,
        input_norm: str | None = None,
        input_norm_eps: float = 1e-5,
        recon_loss: str = "l1",
        huber_delta: float = 1.0,
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
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            input_norm=input_norm,
            input_norm_eps=input_norm_eps,
            recon_loss=recon_loss,
            huber_delta=huber_delta,
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
