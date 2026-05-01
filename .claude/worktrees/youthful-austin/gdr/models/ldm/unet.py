from typing import Sequence, Optional, Callable

from a_unet import TimeConditioningPlugin, ClassifierFreeGuidancePlugin
from a_unet.apex import (
    XUNet,
    XBlock,
    ResnetItem as R,
    AttentionItem as A,
    CrossAttentionItem as C,
    ModulationItem as M,
    SkipCat
)
import torch

from dyno.models.utils.base import BaseModule

from torch import nn

class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


class UNet(BaseModule):

    def __init__(
        self,
        dim: int,
        in_channels: int,
        channels: Sequence[int],
        factors: Sequence[int],
        items: Sequence[int],
        attentions: Sequence[int],
        cross_attentions: Sequence[int],
        attention_features: int,
        attention_heads: int,
        embedding_features: Optional[int] = None,
        skip_t: Callable = SkipCat,
        resnet_groups: int = 8,
        modulation_features: int = 1024,
        embedding_max_length: int = 0,
        use_classifier_free_guidance: bool = False,
        out_channels: Optional[int] = None,
        ckpt_path = None,
        freeze = False,
        **kwargs
    ):
        
        super().__init__(ckpt_path = ckpt_path, freeze = freeze)
        num_layers = len(channels)
        sequences = (channels, factors, items, attentions, cross_attentions)
        assert all(len(sequence) == num_layers for sequence in sequences)
        self.in_channels = in_channels
        # Define UNet type with time conditioning and CFG plugins
        
        self.use_time = kwargs.get("use_time", True)
        unet = TimeConditioningPlugin(XUNet) if self.use_time else XUNet
        
        if use_classifier_free_guidance:
            unet = ClassifierFreeGuidancePlugin(unet, embedding_max_length)
            
        self.unet = unet(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            blocks=[
                XBlock(
                    channels=channels,
                    factor=factor,
                    items=([R, M] + [A] * n_att + [C] * n_cross) * n_items,
                ) for channels, factor, n_items, n_att, n_cross in zip(*sequences)
            ],
            skip_t=skip_t,
            attention_features=attention_features,
            attention_heads=attention_heads,
            embedding_features=embedding_features,
            modulation_features=modulation_features,
            resnet_groups=resnet_groups
        )

        if freeze:
            for param in self.unet.parameters():
                param.requires_grad = False
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def forward(
        self,
        x: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        embedding: Optional[torch.Tensor] = None,
        negative_embedding: Optional[torch.Tensor] = None,
        embedding_scale: float = 1.0,
        embedding_mask_proba: float = 0.0
    ) -> torch.Tensor:
        
        if not self.use_time:
            return self.unet(x)
        
        # if negative_embedding is None:
        #     return self.unet(x,time = time,embedding = embedding, embedding_scale = embedding_scale, embedding_mask_proba = embedding_mask_proba)
        # else:
        #     uncond = self.unet(x,time = time,embedding = embedding, embedding_scale = 0, embedding_mask_proba = embedding_mask_proba)
        #     negative = self.unet(x,time = time,embedding = negative_embedding, embedding_scale = embedding_scale, embedding_mask_proba = embedding_mask_proba)
        #     positive = self.unet(x,time = time,embedding = embedding, embedding_scale = embedding_scale, embedding_mask_proba = embedding_mask_proba)
            
        #     pure_neg = ((negative - uncond) / embedding_scale) + uncond
        #     pure_pos = ((positive - uncond) / embedding_scale) + uncond
            
        #     return pure_neg + embedding_scale * (pure_pos - pure_neg)

        return self.unet(x,time = time,embedding = embedding, embedding_scale = embedding_scale, embedding_mask_proba = embedding_mask_proba)