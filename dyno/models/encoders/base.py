# gdr/models/encoders/base.py
import torch
import torch.nn as nn
from dyno.models.utils.base import BaseModule


class AudioEncoderBase(BaseModule):
    """
    Base class for all audio encoders.

    Subclasses must set:
        output_dim: int
        sample_rate: int
        pool: bool        (True = one vector per chunk; False = sequence)
        downsample_factor: int  (1 = no downsampling; only used when pool=False)
    """
    output_dim: int = 0
    sample_rate: int = 48000
    pool: bool = True
    downsample_factor: int = 1

    def __init__(self, pool: bool = True, downsample_factor: int = 1, ckpt_path=None, freeze: bool = True):
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.pool = pool
        self.downsample_factor = downsample_factor

    def _pool_or_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, D) if pool else (B, T//ds, D)"""
        if self.pool:
            return x.mean(dim=1)
        if self.downsample_factor > 1:
            # adaptive_avg_pool1d expects (B, D, T)
            x = x.permute(0, 2, 1)
            T_out = x.shape[-1] // self.downsample_factor
            x = nn.functional.adaptive_avg_pool1d(x, T_out)
            x = x.permute(0, 2, 1)
        return x

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def extract_features(self, audio: torch.Tensor, return_dict: bool = True, **kwargs):
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        device = next(self.parameters()).device
        audio = audio.to(device)
        features = self.encode(audio)
        if return_dict:
            return {"embedding": features, "embedding_proj": features}
        return features

    def get_audio_embedding_from_data(self, audio, **kwargs):
        return self.extract_features(audio, return_dict=True, **kwargs)
