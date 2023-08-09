# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
from nnunet.network_architecture.attn_unet.my_generic_UNet import *


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DualPathAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0., p2m_feedback=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv_memory = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_pixel = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if p2m_feedback:
            self.proj_p2m = nn.Linear(dim, dim)
            self.proj_p2m_norm = nn.InstanceNorm1d(dim)
            self.proj_p2m_drop = nn.Dropout(proj_drop)
        self.p2m_feedback = p2m_feedback

    def forward(self, pixel_input, memory_input):
        B, N, C = pixel_input.shape
        qkv_pixel = self.qkv_pixel(pixel_input).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3,
                                                                                                              1, 4)
        q_pixel, k_pixel, v_pixel = qkv_pixel.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        B, N, C = memory_input.shape
        qkv_memory = self.qkv_memory(memory_input).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0,
                                                                                                                 3, 1,
                                                                                                                 4)
        q_memory, k_memory, v_memory = qkv_memory.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # q k v shape: (B, H, N, C) Memory attention
        x = self.attend(q_memory, torch.cat((k_pixel, k_memory), dim=2), torch.cat((v_pixel, v_memory), dim=2)).reshape(
            B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # P2M feedback attention
        if self.p2m_feedback:
            y = self.attend(q_pixel, k_memory, v_memory).reshape(B, N, C)
            y = self.proj_p2m(y)
            y = self.proj_p2m_norm(y)
            y = self.proj_drop(y)
            return y, x
        else:
            return x

    def attend(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)  # B N H C
        return x


class DualPathBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, p2m_feedback=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DualPathAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                      p2m_feedback=p2m_feedback)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.p2m_feedback = p2m_feedback

    def forward(self, x, pixel_input):
        pixel_out = None
        if self.p2m_feedback:
            pixel_out, memory_out = self.attn(pixel_input, self.norm1(x))
        else:
            memory_out = self.attn(pixel_input, self.norm1(x))
        x = x + self.drop_path(memory_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.p2m_feedback:
            return pixel_out, x
        else:
            return x
