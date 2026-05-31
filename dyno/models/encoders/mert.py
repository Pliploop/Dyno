# gdr/models/encoders/mert.py
import torch
from .base import AudioEncoderBase


class MERTEncoder(AudioEncoderBase):
    """
    MERT-v1 encoder (m-a-p/MERT-v1-330M).
    Output: sequence at ~75 Hz; output_dim=1024.
    Use layer_index to select a specific transformer layer (default: last).
    """

    output_dim: int = 1024
    sample_rate: int = 24000

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-330M",
        layer_index: int = -1,
        pool: bool = True,
        downsample_factor: int = 1,
        ckpt_path: str = None,
        freeze: bool = True,
        use_safetensors: bool = False,
    ):
        super().__init__(pool=pool, downsample_factor=downsample_factor, ckpt_path=ckpt_path, freeze=freeze)
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=use_safetensors,
        )
        self.model.eval()
        self.layer_index = layer_index
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, samples) at 24kHz
        if audio.ndim == 3:
            audio = audio.mean(dim=1) if audio.shape[1] > 1 else audio.squeeze(1)
        device = audio.device
        # Processor expects numpy; run on CPU then move
        audio_np = audio.cpu().numpy()
        inputs = self.processor(
            audio_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (B, T, 1024), len = 25
        layer = outputs.hidden_states[self.layer_index]  # (B, T, 1024)
        return self._pool_or_downsample(layer)
