"""Adapted from MaskFormer"""
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init


class BasePixelDecoder(nn.Module):
    def __init__(
        self,
        feature_channels: Union[Tuple, List],
        *,
        conv_dim: int,
        mask_dim: int,
        final_stride: int,
        simple_fpn: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
        """
        super().__init__()

        self.feature_channels = feature_channels
        self.final_stride = final_stride
        self.simple_fpn = simple_fpn
        # norm_op = lambda c: nn.GroupNorm(16, c)  # YK: cannot auto cast to fp16, cause OOM
        norm_op = lambda x: nn.InstanceNorm3d(x, affine=True)
        lateral_convs = []
        output_convs = []

        if not simple_fpn:
            use_bias = False
            for idx, in_channels in enumerate(feature_channels):
                if idx == len(feature_channels) - 1:
                    output_norm = norm_op(conv_dim)
                    output_conv = Conv3d(
                        in_channels,
                        conv_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm=output_norm,
                        activation=F.relu,
                    )
                    weight_init.c2_xavier_fill(output_conv)
                    self.add_module("layer_{}".format(idx + 1), output_conv)

                    lateral_convs.append(None)
                    output_convs.append(output_conv)
                else:
                    lateral_norm = norm_op(conv_dim)
                    output_norm = norm_op(conv_dim)

                    # YK: if final_stride is larger than 1, then the last few lateral_conv should have larger stride
                    lateral_stride = {1: 1, 2: [1, 2, 2] if idx == 0 else 1}[final_stride]
                    lateral_conv = Conv3d(
                        in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm, stride=lateral_stride
                    )
                    output_conv = Conv3d(
                        conv_dim,
                        conv_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm=output_norm,
                        activation=F.relu,
                    )
                    weight_init.c2_xavier_fill(lateral_conv)
                    weight_init.c2_xavier_fill(output_conv)
                    self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                    self.add_module("layer_{}".format(idx + 1), output_conv)

                    lateral_convs.append(lateral_conv)
                    output_convs.append(output_conv)
            # Place convs into top-down order (from low to high resolution)
            # to make the top-down computation in forward clearer.
            self.lateral_convs = lateral_convs[::-1]
            self.output_convs = output_convs[::-1]

            self.mask_dim = mask_dim
            self.mask_features = Conv3d(
                conv_dim,
                mask_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            weight_init.c2_xavier_fill(self.mask_features)
        else:
            use_bias = True
            for idx, in_channels in enumerate(feature_channels[:-1]):
                lateral_stride = {1: 1, 2: [1, 2, 2] if idx == 0 else 1}[final_stride]
                lateral_conv = Conv3d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, stride=lateral_stride
                )
                weight_init.c2_xavier_fill(lateral_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)

                lateral_convs.append(lateral_conv)
            output_conv = Conv3d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            weight_init.c2_xavier_fill(output_conv)
            self.output_conv = output_conv
            # Place convs into top-down order (from low to high resolution)
            # to make the top-down computation in forward clearer.
            self.lateral_convs = lateral_convs[::-1]

    def forward(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        features = features[::-1]
        if not self.simple_fpn:
            decoder_fts = []
            for idx, x in enumerate(features):
                lateral_conv = self.lateral_convs[idx]
                output_conv = self.output_convs[idx]
                if lateral_conv is None:
                    y = output_conv(x)
                else:
                    cur_fpn = lateral_conv(x)
                    # features[idx] = cur_fpn  # reduce memory
                    # Following FPN implementation, we use nearest upsampling here
                    upsample_y = y if cur_fpn.shape == y.shape else F.interpolate(y, size=cur_fpn.shape[-3:], mode="nearest")
                    y = cur_fpn + upsample_y
                    y = output_conv(y)
                decoder_fts.append(y)
            return self.mask_features(y), decoder_fts
        else:
            for idx, x in enumerate(features):
                lateral_conv = self.lateral_convs[idx]
                cur_fpn = lateral_conv(x)
                if idx == 0:
                    y = cur_fpn
                else:
                    upsample_y = y if cur_fpn.shape == y.shape else F.interpolate(y, size=cur_fpn.shape[-3:], mode="nearest")
                    y = cur_fpn + upsample_y
            y = self.output_conv(y)
            return y


class Conv3d(nn.Conv3d):
    """
    A wrapper around :class:`torch.nn.Conv3d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            with warnings.catch_warnings(record=True):
                if x.numel() == 0 and self.training:
                    # https://github.com/pytorch/pytorch/issues/12013
                    assert not isinstance(
                        self.norm, torch.nn.SyncBatchNorm
                    ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x