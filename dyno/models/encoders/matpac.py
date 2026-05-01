# gdr/models/encoders/matpac.py
import torch
from .base import AudioEncoderBase


class MatPacEncoder(AudioEncoderBase):
    """
    MatPac encoder (aurianworld/matpac).
    Output: sequence (B, T, 3840) or pooled (B, 3840) depending on pool flag.
    sample_rate: 16000.
    """

    output_dim: int = 3840
    sample_rate: int = 16000

    def __init__(
        self,
        checkpoint_path: str = None,
        pool: bool = True,
        downsample_factor: int = 1,
        ckpt_path: str = None,
        freeze: bool = True,
    ):
        super().__init__(pool=pool, downsample_factor=downsample_factor, ckpt_path=ckpt_path, freeze=freeze)
        try:
            from matpac.model import get_matpac
        except ImportError:
            raise ImportError("Install matpac: pip install matpac (or clone from aurianworld/matpac)")
        self.model = get_matpac(checkpoint_path=checkpoint_path)
        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, samples) at 16kHz
        if audio.ndim == 3:
            audio = audio.mean(dim=1) if audio.shape[1] > 1 else audio.squeeze(1)
        # pull_time_dimension=True → (B, T, 3840); pool mode uses (B, 3840)
        pull_time = not self.pool
        out = self.model(audio, pull_time_dimension=pull_time)
        if isinstance(out, dict):
            seq = out.get("emb", list(out.values())[0])
        else:
            seq = out
        if seq.ndim == 2 and not self.pool:
            seq = seq.unsqueeze(1)  # treat as single-frame sequence
        if seq.ndim == 3:
            return self._pool_or_downsample(seq)
        return seq  # already (B, D)
