# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import PatchEmbed, Mlp
from models.custom_layers import Attention, EqualLinear
from models.pos_embed import VisionRotaryEmbeddingFast
from models.swiglu_ffn import SwiGLUFFN


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        self.latent_embedder = nn.Sequential(
            EqualLinear(hidden_size, hidden_size, lr_mult=0.01), 
            nn.SiLU(),
            EqualLinear(hidden_size, hidden_size, lr_mult=0.01),
        )

    def forward(self, labels, train):
        embeddings = self.embedding_table(labels)
        return self.latent_embedder(embeddings)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, layerscale=1e-1, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"], fused_attn=block_kwargs["fused_attn"],
            )
        
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        use_swiglu = True
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
        
        
        self.ls_attn = nn.Parameter(torch.ones(hidden_size) * layerscale)
        self.ls_mlp = nn.Parameter(torch.ones(hidden_size) * layerscale)


    def forward(self, x, c=None, feat_rope=None):
        
        x = x + self.attn(self.norm1(x), rope=feat_rope) * self.ls_attn
        x = x + self.mlp(self.norm2(x)) * self.ls_mlp
        
        return x

class GATD(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        cmap_dim = 2048,
        **block_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels * 4, hidden_size, bias=True
        )
        
        self.y_embedder = LabelEmbedder(num_classes, cmap_dim, class_dropout_prob)
        self.num_patches = self.x_embedder.num_patches
        
        num_patches = self.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        layer_gain = 1e-1
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, layerscale=layer_gain, **block_kwargs) for _ in range(depth)])
            
        self.final_layer = nn.Sequential(
            nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6),
            nn.Linear(hidden_size, cmap_dim, bias=True),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        self.aux_feat_size = z_dims[0]
        if self.aux_feat_size > 0:
            self.proj = build_mlp(hidden_size, projector_dim, z_dims[0])
        
        self.use_rope = True
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None
            
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
            
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
        
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    
    def forward(self, x, y, t=None, guidance_scale=1.0, return_aux=False):
        y = self.y_embedder(y, self.training)
        y = y.squeeze(dim=1)
            
        x = self.forward_encoder(x, y)
        
        x_cls, x_spatial = x[:, :1], x[:, 1:]
        x_logit = (self.final_layer(x_cls) * y.unsqueeze(1)).sum(-1)
        
        self.recent_x_std = x.std()
        
        if self.aux_feat_size > 0:
            x_feat_spatial = self.proj(x_spatial)
            x_feat_cls = self.proj(x_cls)
            
            x_feat = [x_feat_cls, x_feat_spatial]
        else:
            x_feat = None
        
        if return_aux:
            return_aux = {"x_feat": x_feat}
            return x_logit, return_aux
        
        return x_logit

    def forward_encoder(self, xs, y):
        x = torch.cat([x for x in xs], dim=1)
        
        x = self.x_embedder(x) + self.pos_embed
        N, T, D = x.shape

        cls_token = self.cls_token.repeat([N, 1, 1])
        x = torch.cat([cls_token, x], dim=1)
        
        for i, block in enumerate(self.blocks):
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, y, self.feat_rope, use_reentrant=False)
            
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def GAT_XL_2(**kwargs):
    return GATD(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def GAT_XL_4(**kwargs):
    return GATD(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def GAT_XL_8(**kwargs):
    return GATD(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def GAT_L_2(**kwargs):
    return GATD(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def GAT_L_4(**kwargs):
    return GATD(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def GAT_L_8(**kwargs):
    return GATD(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def GAT_B_2(**kwargs):
    return GATD(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def GAT_B_4(**kwargs):
    return GATD(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def GAT_B_8(**kwargs):
    return GATD(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def GAT_S_2(**kwargs):
    return GATD(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def GAT_S_4(**kwargs):
    return GATD(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def GAT_S_8(**kwargs):
    return GATD(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

GATD_models = {
    'GAT-XL/2': GAT_XL_2,  'GAT-XL/4': GAT_XL_4,  'GAT-XL/8': GAT_XL_8,
    'GAT-L/2':  GAT_L_2,   'GAT-L/4':  GAT_L_4,   'GAT-L/8':  GAT_L_8,
    'GAT-B/2':  GAT_B_2,   'GAT-B/4':  GAT_B_4,   'GAT-B/8':  GAT_B_8,
    'GAT-S/2':  GAT_S_2,   'GAT-S/4':  GAT_S_4,   'GAT-S/8':  GAT_S_8,
}
