# gdr/models/encoders/muq.py
import torch
from .base import AudioEncoderBase


class MuQEncoder(AudioEncoderBase):
    """OpenMuQ / MuQ-MuLan encoder. Produces a single pooled vector per chunk."""

    output_dim: int = 768   # varies by model; overridden after load
    sample_rate: int = 24000
    pool: bool = True
    downsample_factor: int = 1

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-MuLan-large",
        ckpt_path: str = None,
        freeze: bool = True,
    ):
        super().__init__(pool=True, downsample_factor=1, ckpt_path=ckpt_path, freeze=freeze)
        try:
            from muq import MuQ, MuQMuLan
        except ImportError:
            raise ImportError("Install muq: pip install muq")

        self.model_name = model_name
        if "MuLan" in model_name:
            self.muq = MuQMuLan.from_pretrained(model_name)
        else:
            self.muq = MuQ.from_pretrained(model_name)
        self.muq.eval()
        if freeze:
            for param in self.muq.parameters():
                param.requires_grad = False

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 3:
            audio = audio.mean(dim=1) if audio.shape[1] > 1 else audio.squeeze(1)
        elif audio.ndim == 1:
            audio = audio.unsqueeze(0)

        try:
            out = self.muq(wavs=audio)
        except TypeError:
            out = self.muq(audio)

        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
            return h[:, 0] if h.ndim == 3 else h
        raise TypeError(f"Unexpected MuQ output type: {type(out)}")
