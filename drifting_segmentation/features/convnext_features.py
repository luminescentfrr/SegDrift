from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_std(x, dim, eps=1e-6, keepdim=False):
    x = x.float()
    return torch.sqrt(torch.var(x, dim=dim, unbiased=False, keepdim=keepdim).clamp_min(0) + eps)


class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, model_name="convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, freeze=True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("ConvNeXt features require `timm`. Install with `pip install timm`.") from exc
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.freeze = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(False if self.freeze else mode)
        return self

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x + 1) * 0.5
        x = (x - self.mean) / self.std
        feats = self.model(x)
        out = {}
        for i, feat in enumerate(feats):
            b, c, h, w = feat.shape
            normed = feat.float().permute(0, 2, 3, 1)
            normed = (normed - normed.mean(dim=-1, keepdim=True)) / (normed.std(dim=-1, keepdim=True) + 1e-3)
            flat = normed.reshape(b, h * w, c)
            out[f"convnext_stage_{i}"] = flat
            out[f"convnext_stage_{i}_mean"] = flat.mean(dim=1, keepdim=True)
            out[f"convnext_stage_{i}_std"] = safe_std(flat, dim=1, keepdim=True)
        last = out[f"convnext_stage_{len(feats) - 1}"]
        out["global_mean"] = last.mean(dim=1, keepdim=True)
        out["global_std"] = safe_std(last, dim=1, keepdim=True)
        return out
