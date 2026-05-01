import torch
import torch.nn as nn


class DynoBetaVAE(nn.Module):
    """
    Variational bottleneck — reparameterizes CLS token into a stochastic latent z_tau.

    Interface (shared with DynoAutoEncoder):
        forward(h)             → (mu, log_var, z_tau)
        kl_loss(mu, log_var)   → scalar tensor
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.mu_proj = nn.Linear(input_dim, latent_dim)
        self.log_var_proj = nn.Linear(input_dim, latent_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + std * torch.randn_like(std)
        return mu  # deterministic mode at eval (use the posterior mean)

    def kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu_proj(h)
        log_var = self.log_var_proj(h)
        z_tau = self.reparameterize(mu, log_var)
        return mu, log_var, z_tau


class DynoAutoEncoder(nn.Module):
    """
    Deterministic bottleneck — pure AE with no stochasticity or KL penalty.

    Shares the same interface as DynoBetaVAE so the rest of the model is unaffected
    by which bottleneck is configured.
        forward(h)             → (z_tau, None, z_tau)
        kl_loss(mu, log_var)   → 0.0 (always)
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, latent_dim)
        self.act = nn.GELU()

    def kl_loss(self, mu: torch.Tensor, log_var) -> torch.Tensor:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor]:
        z_tau = self.act(self.proj(h))
        return z_tau, None, z_tau   # mu=z_tau, log_var=None — kl_loss will return 0
