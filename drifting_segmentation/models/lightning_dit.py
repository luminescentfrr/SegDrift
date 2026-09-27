import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = torch.einsum("m,d->md", pos.reshape(-1).double(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1)


def apply_rope(q, k):
    bsz, seq_len, heads, dim = q.shape
    half = dim // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    emb = torch.outer(t, freqs)
    emb = torch.cat([emb, emb], dim=-1)
    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]

    def rotate_half(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        use_rmsnorm=False,
        use_rope=False,
        attn_fp32=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.use_rope = use_rope
        self.attn_fp32 = attn_fp32
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.q_norm = norm_cls(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_cls(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x):
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rope(q, k)
        dtype = q.dtype
        if self.attn_fp32:
            q, k, v = q.float(), k.float(), v.float()
        q = q.transpose(1, 2) * (self.head_dim ** -0.5)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, channels)
        return self.proj(out.to(dtype))


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        use_rope=False,
        attn_fp32=True,
    ):
        super().__init__()
        norm_cls = RMSNorm if use_rmsnorm else lambda dim: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1 = norm_cls(hidden_size)
        self.norm2 = norm_cls(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, use_rope=use_rope, attn_fp32=attn_fp32)
        mlp_hidden = int(hidden_size * mlp_ratio)
        if use_swiglu:
            mlp_hidden = int(2 / 3 * mlp_hidden)
            mlp_hidden = (mlp_hidden + 31) // 32 * 32
            self.mlp = SwiGLUFFN(hidden_size, mlp_hidden)
        else:
            self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden_size))
        self.adaLN_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_mod(cond.float()).to(x.dtype).chunk(6, dim=1)
        x = x + gate_msa[:, None, :] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[:, None, :] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN_mod(cond.float()).to(x.dtype).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=1,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_channels=1,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        n_cond_tokens=16,
        attn_fp32=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.n_cond_tokens = n_cond_tokens
        self.use_checkpoint = use_checkpoint
        patch_dim = patch_size * patch_size * in_channels
        self.patch_embed = nn.Linear(patch_dim, hidden_size)
        grid = input_size // patch_size
        pos = get_2d_sincos_pos_embed(hidden_size, grid).unsqueeze(0)
        self.register_buffer("pos_embed", pos, persistent=False)
        self.cond_token_proj = nn.Linear(hidden_size, hidden_size)
        self.cond_token_embed = nn.Parameter(torch.randn(1, n_cond_tokens, hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    use_qknorm,
                    use_swiglu,
                    use_rmsnorm,
                    use_rope,
                    attn_fp32,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, use_rmsnorm)

    def patchify(self, x):
        bsz, channels, height, width = x.shape
        p = self.patch_size
        x = x.reshape(bsz, channels, height // p, p, width // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(bsz, -1, p * p * channels)
        return x

    def unpatchify(self, x):
        bsz, num_patches, _ = x.shape
        p = self.patch_size
        grid = int(num_patches ** 0.5)
        x = x.reshape(bsz, grid, grid, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(bsz, self.out_channels, grid * p, grid * p)
        return x

    def forward(self, x, cond, cond_tokens: Optional[torch.Tensor] = None):
        tokens = self.patch_embed(self.patchify(x)) + self.pos_embed.to(x.device, x.dtype)
        if cond_tokens is None:
            cond_tokens = self.cond_token_proj(cond)[:, None, :].repeat(1, self.n_cond_tokens, 1)
        cond_tokens = cond_tokens + self.cond_token_embed.to(cond_tokens.dtype)
        tokens = torch.cat([cond_tokens, tokens], dim=1)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, cond, use_reentrant=False)
            else:
                tokens = block(tokens, cond)
        tokens = tokens[:, self.n_cond_tokens :, :]
        return self.unpatchify(self.final_layer(tokens, cond))
