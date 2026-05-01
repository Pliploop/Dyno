import torch
import torchvision.ops.stochastic_depth as sd_ops

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy

from .muleblocks import StemModule, NFNetStage, FastToSlowFusion, _scaled_activation
from gdr.models.clap.src.laion_clap.hook import CLAP_Module
    
import torchaudio

from gdr.models.utils.base import BaseModule
    

class Melgram(nn.Module):
    
    def __init__(self, n_mels = 96, n_fft = 2048, window_len = 400, hop_length = 160, sample_rate = 16000, f_min = 0, f_max = 8000, power = 2):
        super(Melgram, self).__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.window_len = window_len
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length = window_len,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
        self.power = power
        
        stype = 'power' if self.power == 2 else 'magnitude'
        self.compressor = torchaudio.transforms.AmplitudeToDB(stype)
        
    def  forward(self, x):
        x = self.mel(x)
        x = self.compressor(x)
        return x    
    
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
default_config = {
    'n_mels': 96,
    'n_fft': 2048,
    'window_len': 400,
    'hop_length': 160,
    'sample_rate': 16000,
    'f_min': 0,
    'f_max': 8000,
    'power': 2
}


class NFNet(BaseModule):
    def __init__(self, frontend = None, f_value = 0, alpha = 0.2, scaled_activation_type = 'gelu', ckpt_path = None, freeze = False):
        super(NFNet, self).__init__(ckpt_path = ckpt_path, freeze = freeze)

        # Initialize parameters for NFNet Blocks
        self.nfnet_stage_depths = [x*(f_value+1) for x in (1,2,6,3)]
        cumulative_stage_depths = np.concatenate(([0],np.cumsum(self.nfnet_stage_depths)))
        self.stoch_depth_survival_probs = 0.1*np.arange(cumulative_stage_depths[-1])/(cumulative_stage_depths[-1])
        self.stoch_depth_survival_probs = [
            self.stoch_depth_survival_probs[st:end] for st, end in zip(cumulative_stage_depths[:-1], cumulative_stage_depths[1:])
        ]
        self.stage_expected_vars = [1.0] + [(1.0+alpha**2)**0.5]*3
        self.stage_downsamples = [1] + [2]*3
        self.scaled_activation = _scaled_activation(scaled_activation_type)
        self.projector_activation = torch.nn.functional.relu

        # Make Stems
        print("Making Stems")
        self.slow_stem = StemModule(
                kernels=[
                    [3, 1],
                    [3, 1],
                    [3, 1],
                    [3, 3]
                ],
                in_channels=[1,16, 32, 64],
                out_channels=[16, 32, 64, 128],
                strides=[
                    [2, 8], # Integrated data striding layer into first convolution
                    [1, 1],
                    [1, 1],
                    [2, 2]
                ],
                # manually added padding for same shape
                padding = [
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 1]
                ]
            )

        self.fast_stem = StemModule(
                kernels=[
                    [3, 3],
                    [3, 3],
                    [3, 3],
                    [3, 3]
                ],
                in_channels=[1,2, 4, 8],
                out_channels=[2, 4, 8, 16],
                strides=[
                    [2, 2], # Integrated data striding layer into first convolution
                    [1, 1],
                    [1, 1],
                    [2, 2]
                ],
                # manually added padding for same shape
                padding = [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1]
                ]
            )

        # Construct NFNet stages for the slow path
        slow_nfnet_kernels = [
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]]
        ]
        slow_nfnet_padding = [
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]]
        ]
        slow_nfnet_input_sizes = [128,256, 512, 1536]
        slow_nfnet_output_sizes = [256,512, 1536, 1536]

        print("Making Slow Layers")
        self.slow_layers = nn.ModuleList([
            NFNetStage(
                kernels=k,
                freq_downsample=f,
                input_channels=i,
                output_channels=o, 
                padding = p,
                group_size=128, 
                alpha=alpha,
                input_expected_var=e,
                stoch_depths=s,
                num_blocks=n
            ) for k, f, i, o, e, s, n,p in zip(
                slow_nfnet_kernels,
                self.stage_downsamples,
                slow_nfnet_input_sizes,
                slow_nfnet_output_sizes,
                self.stage_expected_vars,
                self.stoch_depth_survival_probs,
                self.nfnet_stage_depths,
                slow_nfnet_padding
            )
        ])

        # Construct NFNet stages for the fast path
        fast_nfnet_kernels = [
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]]
        ]
        fast_nfnet_padding = [
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]]
        ]
        fast_nfnet_input_sizes = [16,32, 64, 192]
        fast_nfnet_output_sizes = [32,64, 192, 192]

        print("Making Fast Layers")
        self.fast_layers = nn.ModuleList([
            NFNetStage(
                kernels=k,
                freq_downsample=f,
                input_channels=i,
                output_channels=o, 
                group_size=16, 
                alpha=alpha, 
                input_expected_var=e,
                stoch_depths=s,
                num_blocks=n,
                padding = p
            ) for k, f, i, o, e, s, n,p in zip(
                fast_nfnet_kernels,
                self.stage_downsamples,
                fast_nfnet_input_sizes,
                fast_nfnet_output_sizes,
                self.stage_expected_vars,
                self.stoch_depth_survival_probs,
                self.nfnet_stage_depths,
                fast_nfnet_padding
            )
        ])

        print("Making Fusion Layers")
        # Construct fast-to-slow fusion layers
        self.fusion_layers = nn.ModuleList([
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=16, output_channels=128),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=32, output_channels=256),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=64, output_channels=512),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=192, output_channels=1536)
        ])

        # Construct summarization and aggregation layers at the output
        self.output_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((1, 1))
        ])


        self.frontend = frontend if frontend is not None else Melgram.from_config(default_config)
        
        self.embed_dim = 1728


    def forward(self, x):
        
    
        
        if isinstance(x, dict):
            wav = x['audio']
        else:
            wav = x
        
        if len(wav.shape) == 2: ## batch, time
            wav = wav.unsqueeze(1)
          
        spec = self.frontend(wav) if self.frontend is not None else wav
        
        
        
        slow = self.slow_stem(spec)
        fast = self.fast_stem(spec)


        # For each nfnet_transition module
        for fuse, slw_lyr, fst_lyr in zip(self.fusion_layers, self.slow_layers, self.fast_layers):
            slow = fuse(slow, fast)
            slow = slw_lyr(slow)
            fast = fst_lyr(fast)

        # Apply global average pool and concat
        slow_out = self.scaled_activation(self.output_layers[0](slow))
        fast_out = self.scaled_activation(self.output_layers[1](fast))
        output = torch.cat([slow_out, fast_out], dim=1)
        output = output.view(output.size(0), -1)     

        return output

    @torch.no_grad()
    def extract_features(self, x):
        out = self.forward(x)
        return {
            'embeddings': out
        }
        
        
        
class MULE(BaseModule):
    
    def __init__(self,
                 encoder = None,
                 head_dims = [[1728,1728,512]],
                 temperature = 0.1,
                 feat_extract_head = 0,
                 plusplus = False,
                 ckpt_path = None,
                 freeze = False,
                 **kwargs):
        super(MULE,self).__init__(ckpt_path = ckpt_path, freeze = freeze)
        
        if isinstance(encoder, dict):
            self.encoder = instantiate(encoder)
        else:
            self.encoder = encoder()
        
        self.head_dims = head_dims
        self.encoder_dim = self.encoder.embed_dim if encoder else None
        self.heads = []
        self.plusplus = plusplus
        # if plusplus, the last block of the encoder is parallelized and each heads' input is the output of a different block
        
        for dim in head_dims:
            head = []
            last_dim = self.encoder_dim
            for d in dim:
                head.append(nn.Linear(last_dim,d,bias = False))
                head.append(nn.ReLU())
                last_dim = d
            self.heads.append(nn.Sequential(*head))
            
        self.heads = nn.ModuleList(self.heads)
        self.temperature = temperature
        self.feat_extract_head = feat_extract_head
        
        if isinstance(self.feat_extract_head, list):
            self.embed_dim = self.encoder_dim * len(self.feat_extract_head)
            
        else:
            if self.feat_extract_head == -2:
                self.embed_dim = sum([dim[-1] for dim in self.head_dims])
            elif self.feat_extract_head == -1:
                if not self.plusplus:
                    self.embed_dim = self.encoder_dim
                else:
                    self.embed_dim = self.encoder_dim * len(self.heads)
            elif self.feat_extract_head >= 0:
                self.embed_dim = self.head_dims[self.feat_extract_head]
        
        
        print(f'Embedding dimension: {self.embed_dim}')
        #spwn one loss per head
        
        
    def forward(self,x):
        
        if isinstance(x, dict):
            wav = x['audio']
        else:
            wav = x
                
        encoded = self.encoder(wav)
        
        if self.plusplus:
            projected = [head(encoded[i,...]) for i,head in enumerate(self.heads)]
        else:    
            projected = [head(encoded) for head in self.heads]
        
        return {
            'projected':projected,
            'encoded':encoded,
            "wav":wav,
        }
        
    @torch.no_grad()
    def extract_features(self, x):
        out = self.forward(x)
        
        
        
        
        out['projected_normalized'] = [F.normalize(p, dim=-1) for p in out['projected']]
        
        # extract 0th head by default
        
        
        return {
            'embeddings': out['encoded'],
            'projected': out['projected'][0],
            'projected_normalized': out['projected_normalized'][0]
        }
        
    @classmethod
    def from_pretrained(cls, ckpt_path, device = 'cpu'):
        
        model = cls()
        
        frontend = model.encoder.frontend
        model.encoder.frontend = None
        
        if 's3://' in ckpt_path:
            from s3torchconnector import S3Checkpoint
            checkpoint= S3Checkpoint(region='us-east-1')
            with checkpoint.reader(ckpt_path) as f:
                state_dict = torch.load(f, map_location=device)['state_dict']
                print(f"Model loaded from {ckpt_path}")
        else:
            state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
            print(f"Model loaded from {ckpt_path}")
        
        
        try:
            model.load_state_dict(state_dict)
            print("Loaded full state dict")
            
        except Exception as e:
            print(e) 
        
        model.encoder.frontend = frontend
        
        return model
 
from ..ldm.text_encoders import T5TextEncoder
 
class MuleT5EncoderPair(nn.Module):
    
    def __init__(self, text_encoder ='google-t5/t5-base', audio_encoder_ckpt = None):
        super(MuleT5EncoderPair, self).__init__()
        self.text_encoder = T5TextEncoder(model_name=text_encoder)
        self.audio_encoder = MULE.from_pretrained(audio_encoder_ckpt) if audio_encoder_ckpt is not None else MULE()
        
    def get_text_embedding(self, prompts, **kwargs):
        return self.text_encoder.get_text_embedding(prompts, **kwargs)
    
    def get_audio_embedding_from_data(self, data, **kwargs):
        return self.audio_encoder.extract_features(data, **kwargs)
    
    def extract_features(self, data, **kwargs):
        return self.get_audio_embedding_from_data(data, **kwargs)
    
    def freeze(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = True
            
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
    
class MuleCLAPEncoderPair(nn.Module):
    
    def __init__(self, clap_kws ={}, audio_encoder_ckpt = None, clap_ckpt = None):
        super(MuleT5EncoderPair, self).__init__()
        self.audio_encoder = MULE.from_pretrained(audio_encoder_ckpt) if audio_encoder_ckpt is not None else MULE()
        self.encoder_pair = CLAP_Module(**clap_kws)
        self.encoder_pair.load_ckpt(clap_ckpt, verbose=False)
        self.text_encoder = self.encoder_pair.text_encoder
        
        
    def get_text_embedding(self, prompts, **kwargs):
        return self.text_encoder.get_text_embedding(prompts, **kwargs)
    
    def get_audio_embedding_from_data(self, data, **kwargs):
        return self.audio_encoder.extract_features(data, **kwargs)
    
    def extract_features(self, data, **kwargs):
        return self.get_audio_embedding_from_data(data, **kwargs)
    
    def freeze(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = True
            
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
    