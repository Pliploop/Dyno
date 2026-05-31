"""Annotation-free temporal evaluation callbacks."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from dyno.callbacks.utils import BaseCallback
from dyno.evaluation.structure import cosine_distance_matrix, spearman_from_distance_matrices
from dyno.evaluation.temporal import compute_mspf

log = logging.getLogger(__name__)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.float() / a.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    b = b.float() / b.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    return 1.0 - (a * b).sum(dim=-1)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    x = x[keep]
    y = y[keep]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _r2_score(target: np.ndarray, pred: np.ndarray) -> float:
    keep = np.isfinite(target) & np.isfinite(pred)
    if keep.sum() < 2:
        return float("nan")
    target = target[keep]
    pred = pred[keep]
    denom = float(np.sum((target - target.mean()) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((target - pred) ** 2) / denom)


def _minmax_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.nanmin(x, axis=1, keepdims=True)
    hi = np.nanmax(x, axis=1, keepdims=True)
    scale = hi - lo
    out = np.zeros_like(x, dtype=np.float32)
    np.divide(x - lo, np.maximum(scale, eps), out=out, where=np.isfinite(x))
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if keep.sum() < 4:
        return float("nan")
    xr = _rank(x[keep])
    yr = _rank(y[keep])
    zr = _rank(z[keep])
    design = np.stack([np.ones_like(zr), zr], axis=1)
    beta_x = np.linalg.lstsq(design, xr, rcond=None)[0]
    beta_y = np.linalg.lstsq(design, yr, rcond=None)[0]
    return _pearson(xr - design @ beta_x, yr - design @ beta_y)


def _upper_triangle(a: np.ndarray) -> np.ndarray:
    return a[np.triu_indices_from(a, k=1)]


def _euclidean_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.mean(diff ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d.astype(np.float32)


def _mspf_shape_distance_matrix(mspf: np.ndarray) -> np.ndarray:
    return _euclidean_distance_matrix(_minmax_rows(mspf))


def _mean_mutual_info(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    try:
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return float("nan")
    if x.shape[0] < 8:
        return float("nan")
    x_scaled = StandardScaler().fit_transform(x)
    y_scaled = StandardScaler().fit_transform(y)
    vals = []
    for j in range(y_scaled.shape[1]):
        vals.extend(mutual_info_regression(x_scaled, y_scaled[:, j], random_state=seed).tolist())
    return float(np.mean(vals)) if vals else float("nan")


def _center_and_normalize(feats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    feats = feats.float()
    feats = feats - feats.mean(dim=0, keepdim=True)
    return feats / feats.norm(dim=1, keepdim=True).clamp_min(eps)


def _hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.full((n, n), 1.0 / n, device=K.device, dtype=K.dtype)
    return torch.trace(H @ K @ H @ L) / max((n - 1) ** 2, 1)


def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    if n < 4:
        return _hsic_biased(K, L)
    K = K.clone()
    L = L.clone()
    K.fill_diagonal_(0.0)
    L.fill_diagonal_(0.0)
    term1 = (K * L).sum()
    term2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    term3 = 2.0 * (K.sum(dim=0) * L.sum(dim=0)).sum() / (n - 2)
    return (term1 + term2 - term3) / (n * (n - 3))


def _cknna(
    feats_A: np.ndarray,
    feats_B: np.ndarray,
    topk: int | None = None,
    distance_agnostic: bool = False,
    unbiased: bool = True,
) -> float:
    if feats_A.shape[0] != feats_B.shape[0] or feats_A.shape[0] < 4:
        return float("nan")
    n = feats_A.shape[0]
    if topk is None:
        topk = n - 1
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
    topk = min(topk, n - 1)

    A = _center_and_normalize(torch.from_numpy(np.asarray(feats_A, dtype=np.float32)))
    B = _center_and_normalize(torch.from_numpy(np.asarray(feats_B, dtype=np.float32)))
    K = A @ A.T
    L = B @ B.T

    def similarity(K: torch.Tensor, L: torch.Tensor, topk: int) -> torch.Tensor:
        if unbiased:
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
        mask_K = torch.zeros(n, n, device=K.device, dtype=K.dtype).scatter_(1, topk_K_indices, 1.0)
        mask_L = torch.zeros(n, n, device=K.device, dtype=K.dtype).scatter_(1, topk_L_indices, 1.0)
        mask = mask_K * mask_L

        if distance_agnostic:
            return mask.mean()
        if unbiased:
            return _hsic_unbiased(mask * K, mask * L)
        return _hsic_biased(mask * K, mask * L)

    sim_kl = similarity(K, L, topk)
    sim_kk = similarity(K, K, topk)
    sim_ll = similarity(L, L, topk)
    return float((sim_kl / (torch.sqrt((sim_kk * sim_ll).clamp_min(0.0)) + 1e-6)).item())


def _transform_sequence(seq: torch.Tensor, name: str, partner: torch.Tensor | None = None) -> torch.Tensor:
    T = seq.shape[0]
    if T < 2:
        return seq
    if name == "shuffle":
        return seq[torch.randperm(T, device=seq.device)]
    if name == "reverse":
        return torch.flip(seq, dims=(0,))
    if name == "half_swap":
        half = T // 2
        return torch.cat([seq[half:], seq[:half]], dim=0)
    if name == "circular_shift":
        return torch.roll(seq, shifts=max(1, T // 3), dims=0)
    if name == "local_shuffle":
        out = seq.clone()
        width = max(2, min(T, T // 5))
        start = random.randint(0, T - width)
        perm = torch.randperm(width, device=seq.device)
        out[start : start + width] = out[start : start + width][perm]
        return out
    if name == "splice":
        half = T // 2
        if partner is None or partner.shape[0] < 2:
            return torch.cat([seq[:half], torch.flip(seq[half:], dims=(0,))], dim=0)
        other = partner
        if other.shape[0] < T:
            pad = other[-1:].expand(T - other.shape[0], -1)
            other = torch.cat([other, pad], dim=0)
        return torch.cat([seq[:half], other[:T][half:]], dim=0)
    raise ValueError(f"Unknown transform: {name}")


class AnnotationFreeTemporalCallback(BaseCallback):
    """Evaluate temporal geometry and controlled sequence perturbations.

    Logs scalar metrics as ``{evaluation_suite}/{split}/{metric}``.
    """

    def __init__(
        self,
        n_samples: int | None = None,
        every_n_epochs: int = 5,
        evaluation_suite: str = "AnnotationFreeTemporal",
        mspf_window: int = 30,
        mspf_sigma: float = 10.0,
        mspf_lam: float = 1e-3,
        mspf_power: float = 1.0,
        mspf_points: int = 64,
        transform_agreement: tuple[str, ...] = ("shuffle", "reverse", "splice"),
        order_transforms: tuple[str, ...] = (
            "shuffle",
            "reverse",
            "half_swap",
            "circular_shift",
            "local_shuffle",
        ),
        random_dim: int = 32,
        seed: int = 0,
        estimate_mi: bool = True,
        cknna_topk: int | None = None,
        latent_swapping: bool = True,
    ):
        super().__init__(every_n_epochs=every_n_epochs)
        self.n_samples = n_samples
        self.evaluation_suite = evaluation_suite
        self.mspf_kw = dict(
            window=mspf_window,
            sigma=mspf_sigma,
            lam=mspf_lam,
            power=mspf_power,
            absolute=False,
            n_points=mspf_points,
        )
        self.transform_agreement = tuple(transform_agreement)
        self.order_transforms = tuple(order_transforms)
        self.random_dim = random_dim
        self.seed = seed
        self.estimate_mi = estimate_mi
        self.cknna_topk = cknna_topk
        self.latent_swapping = latent_swapping
        self._samples: dict[str, list[torch.Tensor]] = {"train": [], "val": [], "test": []}
        self._collected: dict[str, bool] = {"train": False, "val": False, "test": False}

    def _reset_stage(self, stage: str) -> None:
        self._samples[stage] = []
        self._collected[stage] = False

    def _collect_batch(self, stage: str, trainer, pl_module, batch) -> None:
        if not self._check_epoch(trainer, pl_module) or self._collected[stage]:
            return

        remaining = None if self.n_samples is None else max(self.n_samples - len(self._samples[stage]), 0)
        if remaining == 0:
            self._collected[stage] = True
            return

        x = batch["audio"].to(device=pl_module.device, non_blocking=True)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device=pl_module.device, dtype=torch.bool, non_blocking=True)

        was_training = pl_module.training
        pl_module.eval()
        with torch.inference_mode():
            x_norm = pl_module.normalize_input(x) if hasattr(pl_module, "normalize_input") else x
            indices = list(range(x_norm.shape[0]))
            random.shuffle(indices)
            for i in indices[:remaining]:
                seq = x_norm[i, mask[i]] if mask is not None else x_norm[i]
                if seq.shape[0] >= 2:
                    self._samples[stage].append(seq.detach().cpu())
        if was_training:
            pl_module.train()

        if self.n_samples is not None and len(self._samples[stage]) >= self.n_samples:
            self._collected[stage] = True
        elif self.n_samples is None and stage == "train":
            self._collected[stage] = True

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset_stage("train")

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset_stage("val")

    def on_test_epoch_start(self, trainer, pl_module):
        self._reset_stage("test")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect_batch("train", trainer, pl_module, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect_batch("val", trainer, pl_module, batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect_batch("test", trainer, pl_module, batch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_stage("train", trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_stage("val", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_stage("test", trainer, pl_module)

    def _model_reps(self, pl_module, seqs: list[torch.Tensor]) -> dict[str, np.ndarray]:
        device = pl_module.device
        lengths = torch.tensor([s.shape[0] for s in seqs], device=device)
        x = pad_sequence([s.to(device=device) for s in seqs], batch_first=True)
        mask = torch.arange(x.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
        with torch.inference_mode():
            zc = (
                pl_module.get_content_token(x, mask=mask)
                if hasattr(pl_module, "get_content_token")
                else _masked_mean(x, mask)
            )
            ztau = pl_module.encode(x, mask=mask)[2]
        zc_cpu = zc.detach().cpu().float()
        ztau_cpu = ztau.detach().cpu().float()
        x_cpu = x.detach().cpu()
        lengths_cpu = lengths.detach().cpu().tolist()
        mspf = [
            compute_mspf(x_cpu[i, : int(length)], **self.mspf_kw)
            for i, length in enumerate(lengths_cpu)
        ]
        rng = np.random.default_rng(self.seed)
        return {
            "Random": rng.standard_normal((len(seqs), self.random_dim), dtype=np.float32),
            "zC": zc_cpu.numpy(),
            "MSPF_feat": np.asarray(mspf, dtype=np.float32),
            "z_tau": ztau_cpu.numpy(),
            "zC_z_tau": torch.cat([zc_cpu, ztau_cpu], dim=-1).numpy(),
        }

    def _log_stage(self, stage: str, trainer, pl_module) -> None:
        seqs = self._samples[stage]
        if not self._check_epoch(trainer, pl_module) or len(seqs) < 2:
            return
        if self.n_samples is not None:
            seqs = seqs[: self.n_samples]
        prefix = f"{self.evaluation_suite}/{stage}"
        metrics: dict[str, float] = {}

        curves = [compute_mspf(seq, **self.mspf_kw) for seq in seqs]
        for transform in self.transform_agreement:
            r2s = []
            for i, seq in enumerate(seqs):
                partner = seqs[(i + 1) % len(seqs)]
                transformed = _transform_sequence(seq, transform, partner=partner)
                curve = compute_mspf(transformed, **self.mspf_kw)
                r2s.append(_r2_score(_minmax_rows(np.asarray([curves[i]]))[0], _minmax_rows(np.asarray([curve]))[0]))
            metrics[f"MSPF R2 after {transform} (transformed vs original)"] = float(np.nanmean(r2s))

        reps = self._model_reps(pl_module, seqs)
        mspf_distance = _mspf_shape_distance_matrix(reps["MSPF_feat"])
        content_distance = cosine_distance_matrix(reps["zC"])
        for name in ("Random", "zC", "z_tau", "zC_z_tau"):
            rep = reps[name]
            rep_distance = cosine_distance_matrix(rep)
            metrics[f"Spearman rho: {name} distances vs MSPF distances"] = spearman_from_distance_matrices(
                rep_distance,
                mspf_distance,
            )
            metrics[f"Partial Spearman rho: {name} distances vs MSPF distances, controlling content"] = _partial_spearman(
                _upper_triangle(rep_distance),
                _upper_triangle(mspf_distance),
                _upper_triangle(content_distance),
            )
            metrics[f"Spearman rho: {name} distances vs content distances"] = spearman_from_distance_matrices(
                rep_distance,
                content_distance,
            )
        if self.estimate_mi:
            metrics["Mutual information: z_tau vs zC"] = _mean_mutual_info(
                reps["zC"],
                reps["z_tau"],
                seed=self.seed,
            )
            metrics["CKNNA: z_tau vs zC"] = _cknna(
                reps["zC"],
                reps["z_tau"],
                topk=self.cknna_topk,
            )

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()
        with torch.inference_mode():
            original_batch = pad_sequence([s.to(device=device) for s in seqs], batch_first=True)
            lengths = torch.tensor([s.shape[0] for s in seqs], device=device)
            mask = torch.arange(original_batch.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
            content_orig = (
                pl_module.get_content_token(original_batch, mask=mask)
                if hasattr(pl_module, "get_content_token")
                else _masked_mean(original_batch, mask)
            )
            temporal_orig = pl_module.encode(original_batch, mask=mask)[2]
            content_orig = content_orig.detach().cpu()
            temporal_orig = temporal_orig.detach().cpu()
            lengths_cpu = lengths.detach().cpu()
            temporal_distance = torch.pdist(temporal_orig.float(), p=2)
            temporal_scale = float(temporal_distance.mean().clamp_min(1e-8).item()) if temporal_distance.numel() else 1.0

            for transform in self.order_transforms:
                transformed = [
                    _transform_sequence(seq.to(device=device), transform, partner=seqs[(i + 1) % len(seqs)].to(device=device))
                    for i, seq in enumerate(seqs)
                ]
                transformed_batch = pad_sequence(transformed, batch_first=True)
                transformed_lengths = torch.tensor([s.shape[0] for s in transformed], device=device)
                transformed_mask = (
                    torch.arange(transformed_batch.shape[1], device=device).unsqueeze(0)
                    < transformed_lengths.unsqueeze(1)
                )
                content_new = (
                    pl_module.get_content_token(transformed_batch, mask=transformed_mask)
                    if hasattr(pl_module, "get_content_token")
                    else _masked_mean(transformed_batch, transformed_mask)
                )
                temporal_new = pl_module.encode(transformed_batch, mask=transformed_mask)[2]
                content_new = content_new.detach().cpu()
                temporal_new = temporal_new.detach().cpu()
                temporal_l2 = torch.linalg.vector_norm(temporal_orig - temporal_new, dim=-1)
                metrics[f"Content cosine distance after {transform} (transformed vs original)"] = float(
                    _cosine_distance(content_orig, content_new).mean().item()
                )
                metrics[f"Temporal cosine distance after {transform} (transformed vs original)"] = float(
                    _cosine_distance(temporal_orig, temporal_new).mean().item()
                )
                metrics[f"Normalized temporal displacement after {transform}"] = float(
                    temporal_l2.mean().item() / temporal_scale
                )

            if self.latent_swapping and len(seqs) >= 2 and hasattr(pl_module, "decode"):
                content_correct = []
                temporal_correct = []
                pair_count = (len(seqs) // 2) * 2
                for a_idx in range(0, pair_count, 2):
                    b_idx = a_idx + 1
                    for temporal_idx, anchor_idx in ((a_idx, b_idx), (b_idx, a_idx)):
                        T = int(lengths_cpu[temporal_idx].item())
                        decoded = pl_module.decode(
                            temporal_orig[temporal_idx : temporal_idx + 1].to(device=device),
                            content_orig[anchor_idx : anchor_idx + 1].to(device=device),
                            T,
                        )
                        decoded_mask = torch.ones(1, T, dtype=torch.bool, device=device)
                        decoded_content = (
                            pl_module.get_content_token(decoded, mask=decoded_mask)
                            if hasattr(pl_module, "get_content_token")
                            else decoded.mean(dim=1)
                        ).detach().cpu()
                        decoded_cpu = decoded.detach().cpu()
                        d_anchor = torch.linalg.vector_norm(decoded_content[0] - content_orig[anchor_idx])
                        d_temporal = torch.linalg.vector_norm(decoded_content[0] - content_orig[temporal_idx])
                        content_correct.append(float((d_anchor < d_temporal).item()))

                        decoded_curve = compute_mspf(decoded_cpu[0], **self.mspf_kw)
                        temporal_curve = curves[temporal_idx]
                        anchor_curve = curves[anchor_idx]
                        temporal_correct.append(
                            float(_rmse(decoded_curve, temporal_curve) < _rmse(decoded_curve, anchor_curve))
                        )
                if content_correct:
                    metrics["Latent swap accuracy: decoded content matches content source"] = float(np.mean(content_correct))
                    metrics["Latent swap accuracy: decoded MSPF matches temporal source"] = float(np.mean(temporal_correct))

        if was_training:
            pl_module.train()

        pl_module.log_dict(
            {f"{prefix}/{key}": value for key, value in metrics.items()},
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self._samples[stage] = []
