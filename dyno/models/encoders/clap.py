import torch
from .base import AudioEncoderBase

dyno
class CLAPEncoder(AudioEncoderBase):
    """
    LAION CLAP encoder via HuggingFace (laion/clap-htsat-fused or compatible).
    Always produces a single pooled 512-dim vector per chunk.
    The processor handles resampling, so any input sample rate is accepted.
    """

    output_dim: int = 512
    sample_rate: int = 48000
    pool: bool = True
    downsample_factor: int = 1

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-fused",
        ckpt_path: str = None,
        freeze: bool = True,
    ):
        super().__init__(pool=True, downsample_factor=1, ckpt_path=ckpt_path, freeze=freeze)
        from transformers import ClapModel, ClapProcessor
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, samples) — processor handles resampling internally
        if audio.ndim == 3:
            audio = audio.mean(dim=1) if audio.shape[1] > 1 else audio.squeeze(1)
        device = audio.device
        audio_np = audio.cpu().numpy()
        inputs = self.processor(
            audios=audio_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return self.model.get_audio_features(**inputs)  # (B, 512)
