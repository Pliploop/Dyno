# gdr/models/encoders/music2latent.py
import torch
import numpy as np
from .base import AudioEncoderBase


class Music2LatentEncoder(AudioEncoderBase):
    """
    Music2Latent encoder (Sony CSL).
    Output: latent sequence at ~10-12 Hz; output_dim=64.
    sample_rate: 44100 or 48000 (pass whichever matches your audio).
    """

    output_dim: int = 64

    def __init__(
        self,
        sample_rate: int = 44100,
        pool: bool = False,
        downsample_factor: int = 1,
        ckpt_path: str = None,
        freeze: bool = True,
    ):
        super().__init__(pool=pool, downsample_factor=downsample_factor, ckpt_path=ckpt_path, freeze=freeze)
        try:
            from music2latent import EncoderDecoder
        except ImportError:
            raise ImportError("Install music2latent: pip install music2latent")
        self.sample_rate = sample_rate
        self.encdec = EncoderDecoder()
        if freeze:
            for param in self.encdec.parameters():
                param.requires_grad = False

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, samples)
        if audio.ndim == 3:
            audio = audio.mean(dim=1) if audio.shape[1] > 1 else audio.squeeze(1)
        # music2latent encode returns (B, 64, T) — needs numpy
        audio_np = audio.cpu().numpy()
        device = audio.device
        latents = self.encdec.encode(audio_np, extract_features=True)  # (B, 64, T)
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).to(device)
        elif isinstance(latents, torch.Tensor):
            latents = latents.to(device)
        # Transpose to (B, T, 64)
        latents = latents.permute(0, 2, 1).float()
        return self._pool_or_downsample(latents)
