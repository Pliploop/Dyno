import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader

from dyno.models.utils.base import BaseModule
from .aggregator import DynoAggregator
from .bottleneck import DynoBetaVAE, DynoAutoEncoder
from .predictor import DynoVelocityPredictor


class DatasetStatsNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
        self.std.copy_(std.to(device=self.std.device, dtype=self.std.dtype).clamp_min(self.eps))
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized.item():
            raise RuntimeError(
                "DatasetStatsNorm has not been initialized. Run trainer.fit so "
                "LightningDyno can compute dataset statistics."
            )
        return (x - self.mean.view(1, 1, -1)) / self.std.view(1, 1, -1)


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

    Conditioning
    ------------
    condition_z_tau and condition_zc control which tokens the predictor receives.
    Disabling condition_zc is valid only for output modes where content is supplied
    outside the predictor: velocity and centered_residuals.

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
        temporal_only_conditioning: bool = False,
        condition_z_tau: bool | None = None,
        condition_zc: bool | None = None,
        embedding_dim: int | None = None,
        model_dim: int | None = None,
        aggregator_model_dim: int | None = None,
        predictor_model_dim: int | None = None,
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
        if temporal_only_conditioning:
            condition_z_tau = True
            condition_zc = False
        if condition_z_tau is None:
            condition_z_tau = getattr(predictor, "condition_z_tau", True)
        if condition_zc is None:
            condition_zc = getattr(predictor, "condition_zc", True)
        if not condition_z_tau and not condition_zc:
            raise ValueError("At least one condition must be enabled: condition_z_tau or condition_zc.")
        predictor_condition_z_tau = getattr(predictor, "condition_z_tau", True)
        predictor_condition_zc = getattr(predictor, "condition_zc", not getattr(predictor, "temporal_only_conditioning", False))
        if predictor_condition_z_tau != condition_z_tau or predictor_condition_zc != condition_zc:
            raise ValueError(
                "Dyno condition_z_tau/condition_zc must match "
                "predictor.condition_z_tau/predictor.condition_zc."
            )
        if not condition_zc and output_mode == "embeddings":
            raise ValueError(
                "condition_zc=False is only available with "
                "output_mode='velocity' or output_mode='centered_residuals'."
            )
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.aggregator = aggregator
        self.bottleneck = bottleneck
        self.predictor = predictor
        self.beta = beta
        self.content_token_mode = content_token
        self.output_mode = output_mode
        self.condition_z_tau = condition_z_tau
        self.condition_zc = condition_zc
        self.temporal_only_conditioning = condition_z_tau and not condition_zc
        self.embedding_dim = embedding_dim
        self.aggregator_model_dim = aggregator_model_dim or model_dim
        self.predictor_model_dim = predictor_model_dim or model_dim
        self.model_dim = model_dim or self.aggregator_model_dim
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
            self.input_norm = nn.LayerNorm(input_dim, eps=input_norm_eps, elementwise_affine=False)
        elif self.input_norm_mode in ("dataset_stats", "dataset", "full_dataset"):
            self.input_norm = DatasetStatsNorm(input_dim, eps=input_norm_eps)
        else:
            raise ValueError("input_norm must be one of: none, identity, layernorm, dataset_stats")

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_norm(x)

    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.recon_loss == "l1":
            return F.l1_loss(x_hat, x)
        if self.recon_loss in ("huber", "smooth_l1"):
            return F.huber_loss(x_hat, x, delta=self.huber_delta)
        return F.mse_loss(x_hat, x)

    def masked_reconstruction_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            return self.reconstruction_loss(x_hat[mask], x[mask])
        return self.reconstruction_loss(x_hat, x)

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
        if self.output_mode == "velocity":
            return x[:, 1:, :] - x[:, :-1, :]
        if self.output_mode == "centered_residuals":
            return x - content.unsqueeze(1)
        return x

    def reconstruction_mask(self, mask: torch.Tensor | None = None) -> torch.Tensor | None:
        if mask is None:
            return None
        if self.output_mode == "velocity":
            return mask[:, 1:] & mask[:, :-1]
        return mask

    def decode(self, z_tau: torch.Tensor, content: torch.Tensor, T: int) -> torch.Tensor:
        prediction = self.predict_target(z_tau, content, T)
        return self.decode_prediction(prediction, content)

    def reconstruct_embeddings(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized input embeddings and reconstructed embeddings.

        This always returns embedding-space tensors, even when the training
        objective predicts velocities or centered residuals internally.
        """
        x = self.normalize_input(x)
        B, T, D = x.shape
        content = self.get_content_token(x, mask=mask)
        mu, log_var, z_tau = self.encode(x, mask=mask)
        x_hat = self.decode(z_tau, content, T)
        return x, x_hat

    def diagnostic_noise_reconstruction_losses(
        self,
        x: torch.Tensor,
        content: torch.Tensor,
        z_tau: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        temporal_noise = torch.randn_like(z_tau)
        content_noise = torch.randn_like(content)

        temporal_noise_recon = self.decode(temporal_noise, content, T)
        content_noise_recon = self.decode(z_tau, content_noise, T)

        temporal_noise_loss = self.masked_reconstruction_loss(temporal_noise_recon, x, mask=mask)
        content_noise_loss = self.masked_reconstruction_loss(content_noise_recon, x, mask=mask)
        return temporal_noise_loss, content_noise_loss

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        x = self.normalize_input(x)
        B, T, D = x.shape
        content = self.get_content_token(x, mask=mask)
        mu, log_var, z_tau = self.encode(x, mask=mask)
        x_hat = self.decode(z_tau, content, T)
        return mu, log_var, z_tau, x_hat

    def compute_loss_components(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.normalize_input(x)
        B, T, D = x.shape
        content = self.get_content_token(x, mask=mask)
        mu, log_var, z_tau = self.encode(x, mask=mask)
        prediction = self.predict_target(z_tau, content, T)
        target = self.reconstruction_target(x, content)
        target_mask = self.reconstruction_mask(mask)
        kl_loss = self.bottleneck.kl_loss(mu, log_var)
        recon_loss = self.masked_reconstruction_loss(prediction, target, mask=target_mask)
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss, x, content, z_tau

    def compute_loss(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_loss, recon_loss, kl_loss, _, _, _ = self.compute_loss_components(x, mask=mask)
        return total_loss, recon_loss, kl_loss

    def diagnostic_reconstruction_metrics(
        self,
        x: torch.Tensor,
        content: torch.Tensor,
        z_tau: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T, D = x.shape
        prediction = self.predict_target(z_tau, content, T)
        x_hat = self.decode_prediction(prediction, content)
        target = self.reconstruction_target(x, content)
        target_mask = self.reconstruction_mask(mask)

        if mask is not None:
            x_valid = x[mask]
            x_hat_valid = x_hat[mask]
        else:
            x_valid = x.reshape(-1, D)
            x_hat_valid = x_hat.reshape(-1, D)

        metrics = {
            "embedding_l1": F.l1_loss(x_hat_valid, x_valid),
            "loss_space_l1": self.masked_reconstruction_loss(prediction, target, mask=target_mask),
            "target_abs": target[target_mask].abs().mean() if target_mask is not None else target.abs().mean(),
            "prediction_abs": prediction[target_mask].abs().mean() if target_mask is not None else prediction.abs().mean(),
            "frame_cosine": F.cosine_similarity(x_hat_valid.float(), x_valid.float(), dim=-1).mean(),
        }
        return metrics


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
        temporal_only_conditioning: bool = False,
        condition_z_tau: bool | None = None,
        condition_zc: bool | None = None,
        embedding_dim: int | None = None,
        model_dim: int | None = None,
        aggregator_model_dim: int | None = None,
        predictor_model_dim: int | None = None,
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
            temporal_only_conditioning=temporal_only_conditioning,
            condition_z_tau=condition_z_tau,
            condition_zc=condition_zc,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            aggregator_model_dim=aggregator_model_dim,
            predictor_model_dim=predictor_model_dim,
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

    @property
    def uses_dataset_stats_norm(self) -> bool:
        return isinstance(self.input_norm, DatasetStatsNorm)

    def _compute_dataset_stats_from_loader(self, dataloader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dim = self.input_norm.mean.numel()
        device = self.device
        total = torch.zeros(input_dim, dtype=torch.float64, device=device)
        total_sq = torch.zeros(input_dim, dtype=torch.float64, device=device)
        count = torch.zeros((), dtype=torch.float64, device=device)
        n_batches = 0

        was_training = self.training
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                n_batches += 1
                x = batch["audio"].to(device=device, dtype=torch.float64, non_blocking=True)
                mask = batch.get("attention_mask")
                if n_batches == 1:
                    rank_zero_info(
                        "dataset_stats first batch: "
                        f"audio_shape={tuple(x.shape)}, mask_present={mask is not None}"
                    )
                if mask is not None:
                    mask = mask.to(device=device, dtype=torch.bool, non_blocking=True)
                    valid = x[mask]
                    if valid.numel() == 0:
                        continue
                    total += valid.sum(dim=0)
                    total_sq += valid.square().sum(dim=0)
                    count += valid.shape[0]
                else:
                    total += x.sum(dim=(0, 1))
                    total_sq += x.square().sum(dim=(0, 1))
                    count += x.shape[0] * x.shape[1]
        if was_training:
            self.train()

        if count.item() <= 0:
            raise RuntimeError("Cannot compute dataset_stats input_norm: no valid training frames were found.")
        mean = total / count
        var = (total_sq / count) - mean.square()
        std = var.clamp_min(0.0).sqrt().clamp_min(self.input_norm.eps)
        rank_zero_info(
            "dataset_stats pass complete: "
            f"batches={n_batches}, frames={int(count.item())}, "
            f"mean_abs_avg={mean.abs().mean().item():.6g}, "
            f"std_avg={std.mean().item():.6g}, "
            f"std_min={std.min().item():.6g}, std_max={std.max().item():.6g}"
        )
        return mean.float(), std.float(), count

    def _dataset_stats_dataloader(self):
        train_loader = self.trainer.datamodule.train_dataloader()
        if not isinstance(train_loader, DataLoader):
            rank_zero_info(
                "dataset_stats using datamodule train_dataloader directly "
                f"({type(train_loader).__name__}); could not rebuild as non-shuffled DataLoader."
            )
            return train_loader
        dataset_len = len(train_loader.dataset) if hasattr(train_loader.dataset, "__len__") else "unknown"
        rank_zero_info(
            "dataset_stats dataloader: "
            f"dataset_len={dataset_len}, batch_size={train_loader.batch_size}, "
            f"num_workers={train_loader.num_workers}, shuffle=False, drop_last=False"
        )
        return DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=train_loader.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=train_loader.pin_memory,
            drop_last=False,
            timeout=train_loader.timeout,
            worker_init_fn=train_loader.worker_init_fn,
            multiprocessing_context=train_loader.multiprocessing_context,
            generator=train_loader.generator,
            prefetch_factor=train_loader.prefetch_factor,
            persistent_workers=train_loader.persistent_workers,
            pin_memory_device=train_loader.pin_memory_device,
        )

    def on_fit_start(self) -> None:
        if not self.uses_dataset_stats_norm or self.input_norm.initialized.item():
            return

        if self.trainer.datamodule is None:
            raise RuntimeError("input_norm=dataset_stats requires a LightningDataModule with a train_dataloader.")

        rank = getattr(self.trainer, "global_rank", 0)
        device = self.device
        input_dim = self.input_norm.mean.numel()

        if rank == 0:
            rank_zero_info(
                "Computing dataset_stats input normalization over the full training dataset. "
                f"dim={input_dim}, eps={self.input_norm.eps}"
            )
            mean, std, count = self._compute_dataset_stats_from_loader(self._dataset_stats_dataloader())
        else:
            mean = torch.zeros(input_dim, dtype=torch.float32, device=device)
            std = torch.ones(input_dim, dtype=torch.float32, device=device)
            count = torch.zeros((), dtype=torch.float64, device=device)

        mean = mean.to(device=device)
        std = std.to(device=device)
        count = count.to(device=device)
        mean = self.trainer.strategy.broadcast(mean, src=0)
        std = self.trainer.strategy.broadcast(std, src=0)
        count = self.trainer.strategy.broadcast(count, src=0)
        if count.item() <= 0:
            raise RuntimeError("Cannot compute dataset_stats input_norm: no valid training frames were found.")
        self.input_norm.set_stats(mean, std)
        rank_zero_info(
            "Initialized dataset_stats input normalization buffers: "
            f"frames={int(count.item())}, "
            f"mean_abs_avg={self.input_norm.mean.abs().mean().item():.6g}, "
            f"std_avg={self.input_norm.std.mean().item():.6g}, "
            f"std_min={self.input_norm.std.min().item():.6g}, "
            f"std_max={self.input_norm.std.max().item():.6g}"
        )

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        x = batch["audio"]                                  # (B, T, D)
        mask = batch.get("attention_mask", None)            # (B, T) bool or None
        total, recon, kl, x_norm, content, z_tau = self.compute_loss_components(x, mask=mask)
        on_step = stage == "train"
        self.log(f"{stage}/loss",     total,          on_step=on_step, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/recon",    recon,          on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/kl",       kl,             on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/beta_kl",  self.beta * kl, on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True)
        with torch.no_grad():
            recon_metrics = self.diagnostic_reconstruction_metrics(
                x_norm,
                content,
                z_tau,
                mask=mask,
            )
        self.log_dict(
            {f"{stage}/{name}": value for name, value in recon_metrics.items()},
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        if self.condition_z_tau and self.condition_zc:
            with torch.no_grad():
                temporal_noise_recon, content_noise_recon = self.diagnostic_noise_reconstruction_losses(
                    x_norm,
                    content,
                    z_tau,
                    mask=mask,
                )
            self.log(
                f"{stage}/recon_temporal_noise",
                temporal_noise_recon,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{stage}/recon_content_noise",
                content_noise_recon,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")
