import numpy as np
import os
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import flgo.utils.fmodule as fmodule

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# class PositionalEncoding

def posemb_sincos_1d(h, dim, temperature: int = 10000, dtype=torch.float32):
    x = torch.arange(h)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(1, dim // 2 + 1, 1) / (dim // 2)
    omega = 1.0 / (temperature ** omega)

    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos()), dim=1)
    return pe.type(dtype)


class Multihead_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()

        latent_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.latent_dim = latent_dim
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.atten = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.patch_emb = nn.Linear(dim, latent_dim * 3, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.patch_emb(x)
        qkv = torch.chunk(x, 3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.atten(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.Lin1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.BN = nn.BatchNorm2d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.Lin2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.Lin1(x)
        x = rearrange(x, 'b n d -> b d n 1')
        x = self.BN(x)
        x = rearrange(x, 'b d n 1 -> b n d')
        # x = self.norm(x)
        # x = nn.BatchNorm1d(x.shape[-2])
        x = self.act(x)
        x = self.drop(x)
        x = self.Lin2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Multihead_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, pool, channels=3, dim_head=64,
                 dropout=0., emb_dropout=0., pos_emb):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert pos_emb in {'pe1d', 'learn'}, 'pos_emb must be either pe1d or learn'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding_1d = posemb_sincos_1d(
            h=num_patches + 1, dim=dim,
        )

        self.pos_embedding_random = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.pos_emb = pos_emb

        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, x):

        x = self.to_patch_embedding(x)

        b, n, _ = x.shape

        if self.pos_emb == 'pe1d':
            pos_emb_1d = repeat(self.pos_embedding_1d, 'n d -> b n d', b=b)
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += pos_emb_1d[:, : n + 1]

        elif self.pos_emb == 'learn':
            pos_emb_random = repeat(self.pos_embedding_random, '1 n d -> b n d', b=b)
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += pos_emb_random[:, : n + 1]

        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.linear_head(x)


class OfficeModel(fmodule.FModule):
    def __init__(self):
        super().__init__()
        self.vit = ViT(
            image_size=256,
            patch_size=32,
            num_classes=10,
            channels=3,
            dim=64,
            depth=6,
            heads=8,
            dim_head=64,
            mlp_dim=128,
            pool='cls' or 'mean',
            dropout=0.0,
            emb_dropout=0.0,
            pos_emb='learn' or 'pe1d'
        )

    def forward(self, x):
        return self.vit(x)

class DomainnetModel(fmodule.FModule):
    def __init__(self):
        super().__init__()
        self.vit = ViT(
            image_size=256,
            patch_size=32,
            num_classes=10,
            channels=3,
            dim=64,
            depth=6,
            heads=8,
            dim_head=64,
            mlp_dim=128,
            pool='cls' or 'mean',
            dropout=0.0,
            emb_dropout=0.0,
            pos_emb='learn' or 'pe1d'
        )

    def forward(self, x):
        return self.vit(x)

class PACSModel(fmodule.FModule):
    def __init__(self):
        super().__init__()
        self.vit = ViT(
            image_size=256,
            patch_size=32,
            num_classes=7,
            channels=3,
            dim=64,
            depth=6,
            heads=8,
            dim_head=64,
            mlp_dim=128,
            pool='cls' or 'mean',
            dropout=0.0,
            emb_dropout=0.0,
            pos_emb='learn' or 'pe1d'
        )

    def forward(self, x):
        return self.vit(x)

model_map = {
    'PACS': PACSModel,
    'office': OfficeModel,
    'domainnet': DomainnetModel,
}


def init_local_module(object):
    pass


def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map[dataset]().to(object.device)