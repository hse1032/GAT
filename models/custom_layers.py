from typing import Type

import numpy as np
import torch
import torch.nn as nn


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mult=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.register_parameter("bias", None)
        self.lr_mult = lr_mult
        self.init_weight(lr_mult=lr_mult)

    def init_weight(self, lr_mult):
        nn.init.xavier_uniform_(self.weight, gain=1 / lr_mult)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        bias = self.bias * self.lr_mult if self.bias is not None else None
        return torch.nn.functional.linear(x, self.weight * self.lr_mult, bias=bias)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.RMSNorm,
        fused_attn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, return_attention=False):
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn and not return_attention:
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            attn = None
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v

        x = x.transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj_drop(self.proj(x))
        if return_attention:
            return x, attn
        return x


class FourierFeature(nn.Module):
    def __init__(self, hidden_size, resolution=16):
        super().__init__()
        self.linear = nn.Linear(2, hidden_size)
        y = torch.linspace(-1, 1, steps=resolution)
        x = torch.linspace(-1, 1, steps=resolution)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).view(1, resolution * resolution, 2)
        self.register_buffer("coords", coords)

    def reset_parameters(self):
        nn.init.uniform_(self.linear.weight, -np.sqrt(9 / 2), np.sqrt(9 / 2))

    def forward(self, x):
        return torch.sin(self.linear(self.coords.to(x.dtype))).repeat(x.shape[0], 1, 1)
