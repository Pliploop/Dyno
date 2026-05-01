from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

import torch


class BaseCallback(Callback):

    def __init__(self, every_n_steps = 1, every_n_epochs = 1, **kwargs):
        super().__init__(**kwargs)
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        
    def _check_step(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        if self.every_n_steps is not None:
            return trainer.global_step % self.every_n_steps == 0
        else:
            return False
        
    def _check_epoch(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        if self.every_n_epochs is not None:
            return trainer.current_epoch % self.every_n_epochs == 0
        else:
            return False

    def _should_run(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        return self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)

    def _should_run_on_validation(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        return self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)

    def _should_run_on_test(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        return self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)



def FD(p, q):
    """
    Compute Fréchet Distance between two multivariate Gaussian distributions.
    
    Args:
        p: Tensor of shape (batch_size, num_channels) - first distribution
        q: Tensor of shape (batch_size, num_channels) - second distribution
    
    Returns:
        Fréchet distance between the two distributions
    """
    try:
        from torchaudio.functional import frechet_distance
    except ImportError:
        raise ImportError(
            "torchaudio is required for FD function. Install with: pip install torchaudio"
        )
    
    # shapes (batch_size, num_channels)
    mu_p = p.mean(0)
    mu_q = q.mean(0)
    
    sigma_p = torch.cov(p.T)
    sigma_q = torch.cov(q.T)
    
    # torchaudio.functional.frechet_distance expects:
    # - mu1, sigma1: mean and covariance of first distribution
    # - mu2, sigma2: mean and covariance of second distribution
    return frechet_distance(mu_p, sigma_p, mu_q, sigma_q)



def match_distribution(source: torch.Tensor, target: torch.Tensor, eps=1e-5):
    """
    Adjust `source` embeddings to have the same mean and covariance as `target`.

    Args:
        source: Tensor of shape (B, d) - embeddings to transform (e.g., test set).
        target: Tensor of shape (B2, d) - reference embeddings (e.g., training set).
        eps: Small constant for numerical stability.

    Returns:
        Transformed source tensor with target mean and covariance.
    """
    # Center source and target
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    # Compute covariance matrices
    source_cov = source_centered.T @ source_centered / (source.shape[0] - 1) + eps * torch.eye(source.shape[1])
    target_cov = target_centered.T @ target_centered / (target.shape[0] - 1) + eps * torch.eye(target.shape[1])

    # Compute sqrt and inverse sqrt via eigen decomposition
    def matrix_sqrt_and_inv(mat):
        eigvals, eigvecs = torch.linalg.eigh(mat)
        sqrt_mat = eigvecs @ torch.diag(torch.sqrt(torch.clamp(eigvals, min=eps))) @ eigvecs.T
        inv_sqrt_mat = eigvecs @ torch.diag(1.0 / torch.sqrt(torch.clamp(eigvals, min=eps))) @ eigvecs.T
        return sqrt_mat, inv_sqrt_mat

    _, source_inv_sqrt = matrix_sqrt_and_inv(source_cov)
    target_sqrt, _ = matrix_sqrt_and_inv(target_cov)

    # Apply whitening + coloring transform
    whitened = source_centered @ source_inv_sqrt
    transformed = whitened @ target_sqrt + target_mean

    return transformed
