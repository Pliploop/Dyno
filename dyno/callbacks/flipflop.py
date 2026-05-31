"""FlipFlop validation callback for Dyno temporal embeddings."""

import logging
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from dyno.callbacks.utils import BaseCallback

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


def _comparison_space(pl_module) -> str:
    mode = getattr(pl_module, "input_norm_mode", "none")
    if mode in (None, "none", "identity"):
        return "raw embedding space"
    return f"{mode} normalized space"


def _cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.float()
    b = b.float()
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    return 1.0 - (a * b).sum(dim=-1)


def _masked_mean_token(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _temporal_token(pl_module, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return pl_module.encode(x, mask=mask)[2]


class FlipFlopCallback(BaseCallback):
    """
    Validation-only callback that swaps chunks within embedding sequences and
    checks whether the temporal bottleneck changes more than the mean content token.

    Extremeness is measured from the resulting permutation:
    sum(abs(permuted_position - original_position)) divided by the displacement
    of a full sequence reversal. It is 0 for unchanged order and approaches 1
    for globally reordered sequences.
    """

    def __init__(
        self,
        n_flips: int = 64,
        every_n_epochs: int = 1,
        chunk_width_min: int = 1,
        chunk_width_max: int = 4,
        distance_min: int = 1,
        distance_max: int = 8,
        n_chunks_min: int = 1,
        n_chunks_max: int = 4,
        max_attempts_per_chunk: int = 64,
        evaluation_suite: str = "FlipFlop",
    ):
        super().__init__(every_n_epochs=every_n_epochs)
        if n_flips <= 0:
            raise ValueError("n_flips must be positive")
        if chunk_width_min <= 0 or chunk_width_max < chunk_width_min:
            raise ValueError("chunk_width_min/max must define a positive range")
        if distance_min <= 0 or distance_max < distance_min:
            raise ValueError("distance_min/max must define a positive range")
        if n_chunks_min <= 0 or n_chunks_max < n_chunks_min:
            raise ValueError("n_chunks_min/max must define a positive range")
        self.n_flips = n_flips
        self.chunk_width_min = chunk_width_min
        self.chunk_width_max = chunk_width_max
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.n_chunks_min = n_chunks_min
        self.n_chunks_max = n_chunks_max
        self.max_attempts_per_chunk = max_attempts_per_chunk
        self.evaluation_suite = evaluation_suite
        self._rows: list[dict[str, float]] = []
        self._collected = False

    def _reset(self):
        self._rows = []
        self._collected = False

    @staticmethod
    def _max_permutation_displacement(T: int) -> float:
        positions = torch.arange(T)
        reversed_positions = torch.arange(T - 1, -1, -1)
        return float((positions - reversed_positions).abs().sum().item())

    def _permutation_extremeness(self, perm: torch.Tensor) -> float:
        T = perm.numel()
        if T < 2:
            return 0.0
        positions = torch.arange(T, device=perm.device)
        displacement = (perm - positions).abs().sum().float()
        max_displacement = max(self._max_permutation_displacement(T), 1.0)
        return float((displacement / max_displacement).clamp(0.0, 1.0).item())

    def _sample_transform(self, T: int, device: torch.device) -> tuple[torch.Tensor, dict[str, float]] | None:
        if T < 2:
            return None

        width_max = min(self.chunk_width_max, T // 2)
        if width_max < self.chunk_width_min:
            return None

        width = random.randint(self.chunk_width_min, width_max)
        feasible_distance_max = min(self.distance_max, (T - width) // width)
        if feasible_distance_max < self.distance_min:
            return None

        distance = random.randint(self.distance_min, feasible_distance_max)
        n_chunks_max = min(self.n_chunks_max, max(1, T // (2 * width)))
        if n_chunks_max < self.n_chunks_min:
            return None

        n_chunks = random.randint(self.n_chunks_min, n_chunks_max)
        offset = width * distance
        perm = torch.arange(T, device=device)
        applied = 0

        for _ in range(n_chunks):
            for _attempt in range(self.max_attempts_per_chunk):
                if offset + width > T:
                    break
                start_a = random.randint(0, T - offset - width)
                start_b = start_a + offset
                a = torch.arange(start_a, start_a + width, device=device)
                b = torch.arange(start_b, start_b + width, device=device)
                tmp = perm[a].clone()
                perm[a] = perm[b]
                perm[b] = tmp
                applied += 1
                break

        if applied == 0:
            return None

        meta = {
            "extremeness": self._permutation_extremeness(perm),
            "chunk_width": float(width),
            "chunk_distance": float(distance),
            "n_chunks": float(n_chunks),
            "applied_chunks": float(applied),
        }
        return perm, meta

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._check_epoch(trainer, pl_module) or self._collected:
            return

        x = batch["audio"]
        mask = batch.get("attention_mask", None)
        device = pl_module.device
        x = x.to(device=device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool, non_blocking=True)

        was_training = pl_module.training
        pl_module.eval()
        originals = []
        flipped_sequences = []
        metas = []

        with torch.inference_mode():
            x_norm = pl_module.normalize_input(x) if hasattr(pl_module, "normalize_input") else x
            B = x_norm.shape[0]
            for _ in range(self.n_flips):
                i = random.randrange(B)
                if mask is not None:
                    valid_mask = mask[i]
                    T_valid = int(valid_mask.sum().item())
                    if T_valid < 2:
                        continue
                    seq = x_norm[i, valid_mask].unsqueeze(0)
                else:
                    T_valid = x_norm.shape[1]
                    seq = x_norm[i : i + 1]

                sampled = self._sample_transform(T_valid, device=seq.device)
                if sampled is None:
                    continue
                perm, meta = sampled
                flipped = seq[:, perm, :]
                originals.append(seq.squeeze(0))
                flipped_sequences.append(flipped.squeeze(0))
                metas.append(meta)

            rows = []
            if originals:
                lengths = torch.tensor([s.shape[0] for s in originals], device=device)
                max_len = int(lengths.max().item())
                row_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                original_batch = pad_sequence(originals, batch_first=True)
                flipped_batch = pad_sequence(flipped_sequences, batch_first=True)
                both = torch.cat([original_batch, flipped_batch], dim=0)
                both_mask = torch.cat([row_mask, row_mask], dim=0)

                content, flipped_content = _masked_mean_token(both, both_mask).chunk(2, dim=0)
                temporal, flipped_temporal = _temporal_token(pl_module, both, mask=both_mask).chunk(2, dim=0)
                content = content.detach().cpu()
                flipped_content = flipped_content.detach().cpu()
                temporal = temporal.detach().cpu()
                flipped_temporal = flipped_temporal.detach().cpu()

                content_cos = _cosine_distance(content, flipped_content)
                temporal_cos = _cosine_distance(temporal, flipped_temporal)
                content_l2 = torch.linalg.vector_norm(content - flipped_content, dim=-1)
                temporal_l2 = torch.linalg.vector_norm(temporal - flipped_temporal, dim=-1)

                for j, meta in enumerate(metas):
                    rows.append({
                        **meta,
                        "content_cosine_distance": float(content_cos[j].item()),
                        "temporal_cosine_distance": float(temporal_cos[j].item()),
                        "content_l2_distance": float(content_l2[j].item()),
                        "temporal_l2_distance": float(temporal_l2[j].item()),
                    })

        if was_training:
            pl_module.train()

        self._rows = rows
        self._collected = True

    def on_validation_epoch_end(self, trainer, pl_module):
        split = "val"
        prefix = f"{self.evaluation_suite}/{split}"
        rows = self._rows
        if not self._check_epoch(trainer, pl_module) or not rows:
            return

        content_cos = np.array([r["content_cosine_distance"] for r in rows], dtype=np.float32)
        temporal_cos = np.array([r["temporal_cosine_distance"] for r in rows], dtype=np.float32)
        content_l2 = np.array([r["content_l2_distance"] for r in rows], dtype=np.float32)
        temporal_l2 = np.array([r["temporal_l2_distance"] for r in rows], dtype=np.float32)
        extremeness = np.array([r["extremeness"] for r in rows], dtype=np.float32)

        pl_module.log(f"{prefix}/Content cosine distance (flipped vs original)", float(content_cos.mean()), on_epoch=True, sync_dist=True)
        pl_module.log(f"{prefix}/Temporal cosine distance (flipped vs original)", float(temporal_cos.mean()), on_epoch=True, sync_dist=True)
        pl_module.log(f"{prefix}/FlipFlop score: temporal minus content cosine distance", float((temporal_cos - content_cos).mean()), on_epoch=True, sync_dist=True)
        pl_module.log(f"{prefix}/Content L2 distance (flipped vs original)", float(content_l2.mean()), on_epoch=True, sync_dist=True)
        pl_module.log(f"{prefix}/Temporal L2 distance (flipped vs original)", float(temporal_l2.mean()), on_epoch=True, sync_dist=True)
        pl_module.log(f"{prefix}/Flip extremeness (normalized permutation displacement)", float(extremeness.mean()), on_epoch=True, sync_dist=True)

        wandb_logger = _get_wandb_logger(trainer)
        if wandb_logger is not None:
            try:
                import plotly.graph_objects as go
                import wandb
            except ImportError:
                log.warning("Skipping FlipFlop Plotly log because plotly or wandb is not installed.")
                self._rows = []
                return

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=extremeness,
                y=content_cos,
                mode="markers",
                name="content mean token",
                marker=dict(color="#2196F3", size=7, opacity=0.75),
            ))
            fig.add_trace(go.Scatter(
                x=extremeness,
                y=temporal_cos,
                mode="markers",
                name="temporal bottleneck token",
                marker=dict(color="#FF5722", size=7, opacity=0.75),
            ))
            fig.update_layout(
                title=f"FlipFlop {split} ({_comparison_space(pl_module)})",
                xaxis_title="flip extremeness from normalized permutation displacement",
                yaxis_title="cosine distance",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            )
            wandb_logger.experiment.log({
                f"{prefix}/Cosine distance vs flip extremeness": wandb.Plotly(fig),
                f"{prefix}/FlipFlop samples": wandb.Table(
                    columns=list(rows[0].keys()),
                    data=[[r[k] for k in rows[0].keys()] for r in rows],
                ),
            })

        self._rows = []
