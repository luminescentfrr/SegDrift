import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightning_dit import LightningDiT, TimestepEmbedder


class SimpleImageConditioner(nn.Module):
    def __init__(self, image_channels=3, cond_dim=768, n_tokens=16):
        super().__init__()
        self.n_tokens = n_tokens
        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, cond_dim // 4, 7, stride=2, padding=3),
            nn.GroupNorm(8, cond_dim // 4),
            nn.SiLU(),
            nn.Conv2d(cond_dim // 4, cond_dim // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, cond_dim // 2),
            nn.SiLU(),
            nn.Conv2d(cond_dim // 2, cond_dim, 3, stride=2, padding=1),
            nn.GroupNorm(16, cond_dim),
            nn.SiLU(),
        )
        self.token_proj = nn.Linear(cond_dim, cond_dim)
        self.global_proj = nn.Linear(cond_dim, cond_dim)

    def forward(self, image):
        feat = self.stem(image)
        pooled = feat.mean(dim=(2, 3))
        cond = self.global_proj(pooled)
        tokens = F.adaptive_avg_pool2d(feat, int(self.n_tokens ** 0.5)).flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        return cond, tokens


class ImageConditionedDriftDiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        image_channels=3,
        noise_channels=1,
        out_channels=1,
        patch_size=16,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        cond_dim=768,
        n_cond_tokens=16,
        noise_classes=64,
        noise_coords=32,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        use_checkpoint=True,
        attn_fp32=True,
    ):
        super().__init__()
        self.noise_classes = noise_classes
        self.noise_coords = noise_coords
        self.conditioner = SimpleImageConditioner(image_channels, cond_dim, n_cond_tokens)
        self.cfg_embedder = TimestepEmbedder(cond_dim)
        self.cfg_norm = nn.LayerNorm(cond_dim)
        self.noise_embeds = nn.ModuleList([nn.Embedding(noise_classes, cond_dim) for _ in range(noise_coords)]) if noise_classes > 0 else nn.ModuleList()
        self.cond_to_hidden = nn.Linear(cond_dim, hidden_size) if cond_dim != hidden_size else nn.Identity()
        self.token_to_hidden = nn.Linear(cond_dim, hidden_size) if cond_dim != hidden_size else nn.Identity()
        self.model = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=noise_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            n_cond_tokens=n_cond_tokens,
            attn_fp32=attn_fp32,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, image, noise_mask, cfg_scale=1.0, noise_labels=None):
        bsz = image.shape[0]
        cond, cond_tokens = self.conditioner(image)
        if not torch.is_tensor(cfg_scale):
            cfg_scale = torch.full((bsz,), float(cfg_scale), device=image.device)
        cfg_scale = cfg_scale.reshape(bsz).to(image.device)
        cond = cond + 0.02 * self.cfg_norm(self.cfg_embedder(cfg_scale))
        if self.noise_classes > 0:
            if noise_labels is None:
                noise_labels = torch.randint(self.noise_classes, (bsz, self.noise_coords), device=image.device)
            for i, embed in enumerate(self.noise_embeds):
                cond = cond + embed(noise_labels[:, i])
        return self.model(noise_mask, self.cond_to_hidden(cond), self.token_to_hidden(cond_tokens))
