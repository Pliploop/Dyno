import torch
import torchvision.ops.stochastic_depth as sd_ops

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy

def weight_standardization(weight: torch.Tensor, eps: float):
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / (torch.sqrt(var + eps))
    return weight.view(c_out, c_in, *kernel_shape)



def _scaled_activation(activation_name):
    activations = {
        'gelu': lambda x: torch.nn.functional.gelu(x) * 1.7015043497085571,
        'relu': lambda x: torch.nn.functional.relu(x) * 1.7139588594436646 
    }
    return activations[activation_name]


class StemModule(nn.Module):
    """Create the stem module. This is a series of convolutional layers that are applied on
    the input, prior to any residual stages."""

    def __init__(self, kernels, in_channels, out_channels, strides, padding,activation=F.relu):
        super(StemModule, self).__init__()
        self.layers = self._make_stem_module(kernels, in_channels, out_channels, strides, padding, activation)
        
        print(f"StemModule: {in_channels} -> {out_channels}")

    def _make_stem_module(self, kernels, in_channels,out_channels, strides,padding,activation):
        """Constructs the layers for the stem module."""
        layers = []
        for i,c, k, s,p in zip(in_channels, out_channels, kernels, strides,padding):
            layers.append(nn.Conv2d(i, c, k, stride=s, padding=p))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        """Applies the stem module to an input."""
        x = self.layers(x)
        return x
    
    
class WSConv2D(nn.Module):
    """Creates the variance preserving weight standardized convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=F.relu, padding=(0,0)):
        super(WSConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=False, padding=padding)
        self.activation = activation
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        weight = self.conv.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight_var = weight.var(dim=(1, 2, 3), keepdim=True)
        fan_in = np.prod(weight.shape[1:])
        scale = torch.rsqrt(torch.clamp(weight_var * fan_in, min=1e-4))
        shift = weight_mean * scale
        weight = weight * scale - shift
        x = F.conv2d(x, weight, None, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
    
class SqueezeExcite(nn.Module):
    """Create a squeeze and excite module."""

    def __init__(self, output_channels):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(output_channels, output_channels // 2, bias=True),
            nn.ReLU(),
            nn.Linear(output_channels // 2, output_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    

class FastToSlowFusion(nn.Module):
    """Make layers that comprise the operations in order to fuse the fast path of NFNet stages to the slow path."""

    def __init__(self, time_kernel_length, time_stride, input_channels, output_channels):
        super(FastToSlowFusion, self).__init__()
        self.conv1 = WSConv2D(input_channels, input_channels, kernel_size=(1, time_kernel_length), stride=(1, time_stride))
        self.conv2 = WSConv2D(input_channels, output_channels, kernel_size=1, stride=1)

    def forward(self, slow, fast):
        fast = self.conv1(fast)
        fast = self.conv2(fast)
        return torch.cat([slow, fast], dim=-1)
    
class NFNetBlock(nn.Module):
    def __init__(self, kernels, freq_downsample, input_channels, output_channels, group_size, alpha, beta, stoch_depth, padding,is_transition=False):
        super(NFNetBlock, self).__init__()
        self.is_transition_block = (freq_downsample > 1) or (input_channels != output_channels) or is_transition

        
        if self.is_transition_block:
            self.input_layers = nn.Sequential(
                nn.ReLU(),
                ScalarMultiply(beta)
            )
            self.residual_path = nn.ModuleList()
        else:
            self.input_layers = nn.Identity()
            self.residual_path = nn.Sequential(
                nn.ReLU(),
                ScalarMultiply(beta)
            )

        print(f"NFNetBlock: {input_channels} -> {output_channels} | freq_downsample: {freq_downsample} | is_transition: {self.is_transition_block}")

        strides = [[1, 1], [freq_downsample, 1], [1, 1], [1, 1]]
        per_layer_out_chans = [output_channels // 2] * 3 + [output_channels]
        per_layer_in_chans = [input_channels] + per_layer_out_chans[:-1]
        groups = [1] + [output_channels // 2 // group_size] * 2 + [1]
        activations = [nn.ReLU()] * (len(kernels) - 1) + [None]

        for i,c, k, s, g, a,p in zip(per_layer_in_chans,per_layer_out_chans, kernels, strides, groups, activations, padding):
            self.residual_path.extend([
                WSConv2D(i,c, kernel_size=k, stride=s, groups=g, activation=a, padding = p),
            ])

        self.residual_path.extend([
            SqueezeExcite(per_layer_out_chans[-1]),
            ScalarMultiply(0.0, learnable=True),
            ScalarMultiply(alpha)
        ])
        
        self.residual_path = nn.Sequential(*self.residual_path)

        self.skip_path = nn.Identity()
        if freq_downsample > 1:
            self.skip_path = nn.AvgPool2d(kernel_size=[freq_downsample, 1], stride=[freq_downsample, 1], padding=[freq_downsample//2-1, 0])

        if self.is_transition_block:
            self.skip_path = nn.Sequential(
                self.skip_path,
                WSConv2D(input_channels, output_channels, kernel_size=[1, 1], stride=[1, 1], activation=None, padding = 'same')
            )

        self.output_layers = nn.Sequential(
            StochDepth(survival_probability=1 - stoch_depth, scale_during_test=False)
        )

    def forward(self, x):
        x = self.input_layers(x)
        residual = self.residual_path(x)
        skip = self.skip_path(x)
        output = self.output_layers([skip, residual])
        return output


class StochDepth(nn.Module):
    def __init__(self, survival_probability=0.5, scale_during_test=False):
        super(StochDepth, self).__init__()
        self.survival_probability = survival_probability
        self.scale_during_test = scale_during_test

    def forward(self, x):
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("input must be a list of length 2.")

        shortcut, residual = x

        # Random bernoulli variable indicating whether the branch should be kept or not
        b_l = torch.bernoulli(torch.tensor([self.survival_probability])).to(residual.device)
        b_l = b_l.view(-1, 1, 1, 1)

        if self.training:
            return shortcut + b_l * residual
        else:
            if self.scale_during_test:
                return shortcut + self.survival_probability * residual
            else:
                return shortcut + residual

class ScalarMultiply(nn.Module):
    def __init__(self, scalar, learnable=False):
        super(ScalarMultiply, self).__init__()
        self.scalar = nn.Parameter(torch.tensor(scalar), requires_grad=learnable)

    def forward(self, x):
        return x * self.scalar




class ParallelModule(nn.Module):
    
    def __init__(self, module, num_parallel = None):
        super(ParallelModule, self).__init__()
        if isinstance(module, nn.Module):
            self.parallels = nn.ModuleList([copy.deepcopy(module) for _ in range(num_parallel)])
        elif isinstance(module, list):
            self.parallels = nn.ModuleList(module)
        
    def forward(self,x):
        
        outputs = []
        
        if isinstance(x, torch.Tensor):
            x = [x for _ in range(len(self.parallels))]
        
        for i, module in enumerate(self.parallels):
            outputs.append(module(x[i]))
            
            
        # print(f'ParallelModule: {len(self.parallels)} branches with shapes {[o.shape for o in outputs]}')
        return outputs
            

class NFNetStage(nn.Module):
    def __init__(self, kernels, freq_downsample, input_channels, output_channels, group_size, alpha, input_expected_var, stoch_depths, num_blocks, padding):
        super(NFNetStage, self).__init__()
        self.blocks = nn.ModuleList()

        # NFNet transition block first
        self.blocks.append(NFNetBlock(
            kernels, 
            freq_downsample, 
            input_channels, 
            output_channels,
            group_size, 
            alpha,
            1.0/input_expected_var,
            float(stoch_depths[0]),
            padding,
            is_transition=True
        ))

        # NFNet non-transition blocks
        expected_std = (input_expected_var**2.0 + alpha**2.0)**0.5
        for idx in range(1,num_blocks):
            self.blocks.append(NFNetBlock(
                kernels,
                1, 
                output_channels, 
                output_channels,
                group_size, 
                alpha, 
                1.0/expected_std,
                float(stoch_depths[idx]),
                padding
            ))
            expected_std = (expected_std**2.0 + alpha**2.0)**0.5

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
