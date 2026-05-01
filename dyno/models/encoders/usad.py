# gdr/models/encoders/usad.py
import torch
from .base import AudioEncoderBase


class USADEncoder(AudioEncoderBase):
    """
    USAD-Large encoder (MIT-SLS/USAD-Large).
    Output: sequence at ~50 Hz; output_dim=1024.
    """

    output_dim: int = 1024
    sample_rate: int = 16000

    def __init__(
        self,
        model_name: str = "MIT-SLS/USAD-Large",
        pool: bool = False,
        downsample_factor: int = 1,
        ckpt_path: str = None,
        freeze: bool = True,
    ):
        super().__init__(pool=pool, downsample_factor=downsample_factor, ckpt_path=ckpt_path, freeze=freeze)
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, samples) at 16kHz
        if audio.ndim == 3:
            audio = audio.mean(dim=1) if audio.shape[1] > 1 else audio.squeeze(1)
        outputs = self.model(audio)
        # Returns dict with 'x': (B, T, 1024)
        if isinstance(outputs, dict):
            seq = outputs.get("x", outputs.get("last_hidden_state"))
        elif hasattr(outputs, "last_hidden_state"):
            seq = outputs.last_hidden_state
        else:
            raise TypeError(f"Unexpected USAD output: {type(outputs)}")
        return self._pool_or_downsample(seq)
