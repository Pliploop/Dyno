"""
Temporal evaluation metrics for audio embedding sequences.

MSPF — Music Semantic Progress Function
    A 1D cumulative trajectory of semantic shift across a sequence.
    Adapted from Metzer et al. (2026): "Video Analysis and Generation via a
    Semantic Progress Function" (SIGGRAPH 2026).

SSM  — Self-Similarity Matrix
    Pairwise cosine similarities between all frame pairs in a sequence.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def compute_mspf(
    z: torch.Tensor,
    window: int = 30,
    sigma: float = 10.0,
    lam: float = 1e-3,
    power: float = 1.0,
    absolute: bool = True,
    n_points: int = 100,
) -> np.ndarray:
    """
    Compute the Music Semantic Progress Function for a sequence of embeddings.

    Solves the regularised weighted least-squares problem:
        min_S (AS - b)ᵀ W (AS - b) + λ ‖S‖²
    where each row of A encodes a pair constraint S_j - S_i ≈ d̃_ij,
    b_k = arccos(zᵢ · z_j)^power, and W is a diagonal Gaussian weight
    w_ij = exp(-(j-i)² / 2σ²).

    Args:
        z:        (T, D) tensor of embeddings (will be L2-normalised internally)
        window:   maximum frame-distance to include in pairs (default 30)
        sigma:    temporal Gaussian bandwidth in frames (default 10)
        lam:      L2 regularisation weight (default 1e-3)
        power:    angular distance exponent p; p>1 boosts contrast (default 1)
        absolute: when True (default), return S at original resolution — the
                  x-axis is in frames, so the same song time-stretched gives
                  different SPF curves.  When False, interpolate S onto
                  ``n_points`` uniformly-spaced points on a normalised [0, 1]
                  time axis, making SPFs comparable across sequences of
                  different lengths or tempos.
        n_points: number of output points when absolute=False (default 100)

    Returns:
        absolute=True  → float32 array of shape (T,), anchored at S[0] = 0
        absolute=False → float32 array of shape (n_points,), time axis [0, 1]
    """
    T = z.shape[0]
    if T < 2:
        out_len = T if absolute else n_points
        return np.zeros(out_len, dtype=np.float32)

    z_np = F.normalize(z.float(), dim=-1).detach().cpu().numpy()  # (T, D)

    # Build pair constraints
    pairs = [
        (i, j)
        for i in range(T)
        for j in range(i + 1, min(i + window + 1, T))
    ]
    n_pairs = len(pairs)
    A = np.zeros((n_pairs, T), dtype=np.float32)
    b = np.zeros(n_pairs, dtype=np.float32)
    w = np.zeros(n_pairs, dtype=np.float32)

    for k, (i, j) in enumerate(pairs):
        cos = float(np.clip(np.dot(z_np[i], z_np[j]), -1 + 1e-7, 1 - 1e-7))
        d = math.acos(cos)
        A[k, i] = -1.0
        A[k, j] = 1.0
        b[k] = d ** power
        w[k] = math.exp(-(j - i) ** 2 / (2 * sigma ** 2))

    W_diag = np.diag(w)
    AtW = A.T @ W_diag
    lhs = AtW @ A + lam * np.eye(T, dtype=np.float32)
    rhs = AtW @ b
    S = np.linalg.solve(lhs, rhs).astype(np.float32)
    S -= S[0]  # anchor: S[0] = 0

    if absolute:
        return S

    # Relative mode: interpolate onto a fixed-length normalized time grid
    t_orig = np.linspace(0.0, 1.0, T)
    t_norm = np.linspace(0.0, 1.0, n_points)
    return np.interp(t_norm, t_orig, S).astype(np.float32)


def compute_ssm(z: torch.Tensor) -> np.ndarray:
    """
    Compute the cosine Self-Similarity Matrix for a sequence.

    Args:
        z: (T, D) tensor of embeddings

    Returns:
        SSM: (T, T) numpy array with values in [-1, 1]
    """
    z_np = F.normalize(z.float(), dim=-1).detach().cpu().numpy()
    return (z_np @ z_np.T).astype(np.float32)


def linearity_score(S: np.ndarray) -> float:
    """
    Quantify how closely the MSPF follows an ideal linear pace.

    Score ∈ [0, 1]: 1 = perfectly linear, 0 = maximally non-linear.
    Computed as 1 minus the RMS deviation from the ideal line, normalised
    by the total dynamic range of S.
    """
    T = len(S)
    if T < 2:
        return 1.0
    ideal = np.linspace(S[0], S[-1], T)
    rms = np.sqrt(np.mean((S - ideal) ** 2))
    scale = abs(float(S[-1]) - float(S[0])) + 1e-8
    return float(np.clip(1.0 - rms / scale, 0.0, 1.0))
