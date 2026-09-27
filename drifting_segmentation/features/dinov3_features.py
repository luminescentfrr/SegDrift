from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_std(x, dim, eps=1e-6, keepdim=False):
    x = x.float()
    return torch.sqrt(torch.var(x, dim=dim, unbiased=False, keepdim=keepdim).clamp_min(0) + eps)


class DINOv3FeatureExtractor(nn.Module):
    """DINOv3 ViT feature extractor for drifting loss.

    Uses timm first. The expected model name is something like
    `vit_base_patch16_dinov3.lvd1689m`.
    """

    def __init__(
        self,
        model_name="vit_base_patch16_dinov3.lvd1689m",
        pretrained=True,
        freeze=True,
        image_size=224,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("DINOv3 features require `timm`. Install with `pip install timm`.") from exc
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.freeze = freeze
        self.image_size = image_size
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(False if self.freeze else mode)
        return self

    def _forward_tokens(self, x):
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patch = feats["x_norm_patchtokens"]
                cls = feats.get("x_norm_clstoken")
                return patch, cls
            if "tokens" in feats:
                feats = feats["tokens"]
            elif "last_hidden_state" in feats:
                feats = feats["last_hidden_state"]
            else:
                feats = next(v for v in feats.values() if torch.is_tensor(v) and v.ndim == 3)
        if feats.ndim == 2:
            return feats[:, None, :], feats
        # Strip ALL prefix tokens (CLS + any register tokens).
        # DINOv3 uses num_prefix_tokens=5 (1 CLS + 4 registers).
        # Using [:, 1:] would leave register tokens in the patch sequence,
        # causing a shape mismatch when interpolating the mask to the grid.
        n_prefix = int(getattr(self.model, "num_prefix_tokens", 1) or 1)
        cls = feats[:, 0]
        patch = feats[:, n_prefix:]
        return patch, cls

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x + 1) * 0.5
        x = (x - self.mean) / self.std
        patch, cls = self._forward_tokens(x)
        patch = patch.float()
        out = {
            "dinov3_patch": patch,
            "dinov3_mean": patch.mean(dim=1, keepdim=True),
            "dinov3_std": safe_std(patch, dim=1, keepdim=True),
        }
        if cls is not None:
            out["dinov3_cls"] = cls.float().unsqueeze(1)
        return out

    def forward_weighted(self, image, mask_prob) -> Dict[str, torch.Tensor]:
        """C2: run DINOv3 on the full image, then weight patch tokens by mask in
        feature space.  This preserves ViT global attention (no zeroed patches)
        while still separating foreground from background representations.

        Args:
            image:     [B, 3, H, W] in [-1, 1]
            mask_prob: [B, 1, H, W] in [0, 1]  (soft or hard mask)
        """
        img = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        img = (img + 1) * 0.5
        img = (img - self.mean) / self.std
        patch, cls = self._forward_tokens(img)
        patch = patch.float()

        # compute grid size from model geometry rather than token count,
        # so the result is always an exact integer regardless of prefix tokens
        patch_size = getattr(self.model, "patch_size", None) or getattr(
            self.model, "patch_embed", None
        )
        if patch_size is None or not isinstance(patch_size, int):
            # fallback: infer from token count (safe after prefix stripping)
            n_tokens = patch.shape[1]
            grid = int(n_tokens ** 0.5)
            assert grid * grid == n_tokens, (
                f"patch token count {n_tokens} is not a perfect square; "
                "cannot infer grid size"
            )
        else:
            grid = self.image_size // patch_size

        # interpolate mask to patch grid, then flatten to token weights [B, N, 1]
        weights = F.interpolate(mask_prob.float(), size=(grid, grid), mode="bilinear", align_corners=False)
        weights = weights.flatten(2).transpose(1, 2)          # [B, N, 1]
        weighted_patch = patch * weights                       # [B, N, D]

        out = {
            "dinov3_patch": weighted_patch,
            "dinov3_mean": weighted_patch.mean(dim=1, keepdim=True),
            "dinov3_std": safe_std(weighted_patch, dim=1, keepdim=True),
        }
        if cls is not None:
            out["dinov3_cls"] = cls.float().unsqueeze(1)
        return out
