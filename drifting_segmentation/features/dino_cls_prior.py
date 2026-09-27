import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from drifting_segmentation.losses.segmentation import dice_loss


class ResidualAdapter(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def sigmoid_focal_loss(logits, target, alpha=0.25, gamma=2.0):
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


class DINOCLSCalibrationPrior(nn.Module):
    """Calibrate DINOv3 CLS as a lesion query via CLS-patch attention."""

    def __init__(
        self,
        model_name="vit_base_patch16_dinov3.lvd1689m",
        pretrained=True,
        image_size=224,
        adapter_hidden=1024,
        temperature=0.07,
        focal_alpha=0.25,
        focal_gamma=2.0,
        freeze_dino=True,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("DINO CLS prior requires `timm`. Install with `pip install timm`.") from exc
        self.dino = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.image_size = image_size
        self.temperature = temperature
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.freeze_dino = freeze_dino
        if freeze_dino:
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad_(False)
        embed_dim = getattr(self.dino, "embed_dim", None)
        if embed_dim is None:
            embed_dim = self._infer_embed_dim()
        self.cls_adapter = ResidualAdapter(embed_dim, adapter_hidden)
        self.patch_adapter = ResidualAdapter(embed_dim, adapter_hidden)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _infer_embed_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            tokens = self.dino.forward_features(dummy)
            if isinstance(tokens, dict):
                if "x_norm_clstoken" in tokens:
                    return tokens["x_norm_clstoken"].shape[-1]
                tokens = next(v for v in tokens.values() if torch.is_tensor(v) and v.ndim == 3)
            return tokens.shape[-1]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_dino:
            self.dino.eval()
        return self

    def _tokens(self, image):
        x = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x + 1) * 0.5
        x = (x - self.mean) / self.std
        if self.freeze_dino:
            with torch.no_grad():
                tokens = self.dino.forward_features(x)
        else:
            tokens = self.dino.forward_features(x)
        if isinstance(tokens, dict):
            if "x_norm_patchtokens" in tokens:
                patch = tokens["x_norm_patchtokens"]
                cls = tokens["x_norm_clstoken"]
                return cls, patch
            tokens = next(v for v in tokens.values() if torch.is_tensor(v) and v.ndim == 3)
        num_prefix = int(getattr(self.dino, "num_prefix_tokens", 1) or 1)
        cls = tokens[:, 0]
        patch = tokens[:, num_prefix:]
        return cls, patch

    def forward(self, image, gt_mask=None):
        cls, patch = self._tokens(image)
        bsz, tokens, dim = patch.shape
        grid = int(math.sqrt(tokens))
        if grid * grid != tokens:
            raise RuntimeError(f"Patch token count {tokens} is not a square grid")
        cls_q = F.normalize(self.cls_adapter(cls.float()), dim=-1)
        patch_k = F.normalize(self.patch_adapter(patch.float()), dim=-1)
        logits_patch = torch.einsum("btd,bd->bt", patch_k, cls_q) / self.temperature
        prior_logits = logits_patch.reshape(bsz, 1, grid, grid)
        prior_logits = F.interpolate(prior_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        prior_prob = torch.sigmoid(prior_logits)
        out = {
            "prior_logits": prior_logits,
            "prior_prob": prior_prob,
            "cls_token": cls.detach(),
            "patch_tokens": patch.detach(),
        }
        if gt_mask is not None:
            focal = sigmoid_focal_loss(prior_logits, gt_mask, self.focal_alpha, self.focal_gamma)
            dice = dice_loss(prior_prob, gt_mask)
            out["loss_cls_prior"] = focal + dice
            out["loss_cls_focal"] = focal.detach()
            out["loss_cls_dice"] = dice.detach()
        return out
