# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import models.vgg_ as models

from typing import List, Iterable
import time
from einops import rearrange
from torchvision.ops import StochasticDepth

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
            # backbone = models.vgg16_bn(pretrained=False)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)
        
        
class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x) # (1,16,64,64) -> ( 64 - 8  + 0 ) / 8 + 1 = 8; -> (1,16,8,8)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c") # (1,16,8,8) -> (1, (8*8), 16)
        x = rearrange(x, "b c h w -> b (h w) c") # (1,16,64,64) -> (1, (64 * 64), 16)
        # query, key, value input to the attention layer
        # out = self.att(x, reduced_x, reduced_x) # [0](Attention output) -> (1, 4096, 16); [1](Attention output weights) -> (1, 4096, 64)
        out = self.att(x, reduced_x, reduced_x)[0]
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w) # (1,16,64,64)
        
        return out

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c") # (1, 224, 224, 3)
        x = super().forward(x) # (1, 224, 224, 3)
        x = rearrange(x, "b h w c -> b c h w") # (1, 3, 224, 224)
        return x
    
class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,            # 3
                out_channels,           # 64 
                kernel_size=patch_size, # 16
                stride=overlap_size,    # 16
                padding=patch_size // 2,# 8
                bias=False
            ),
            LayerNorm2d(out_channels)
        )

class ResidualAdd(nn.Module):
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )


class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        
        self.blocks = nn.Sequential(
                *[
                    SegFormerEncoderBlock(
                        out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                    )
                    for i in range(depth)
                ]
            )
        self.norm = LayerNorm2d(out_channels)
        
        

def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk
        
class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )
        
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def build_backbone(args):
    if args.backbone == "vgg16_bn":
        backbone = Backbone_VGG(args.backbone, True)
        print(f"{args.backbone} is created!")
    
    elif args.backbone == "new":
        backbone = SegFormerEncoder(in_channels=3,
                         widths=[128, 256, 512, 512],
                         depths=[3, 4, 3, 3],
                         all_num_heads=[1, 2, 2, 2],
                         patch_sizes=[7, 3, 3, 3],
                         overlap_sizes=[2, 2, 2 , 2],
                         reduction_ratios=[8, 4, 4, 4],
                         mlp_expansions=[4, 4, 4, 4],
                         drop_prob=0)
        print(f"{args.backbone} is created!")
        
    return backbone

if __name__ == '__main__':
    Backbone_VGG('vgg16', True)
