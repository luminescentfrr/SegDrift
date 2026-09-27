from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_gn_groups(channels, max_groups=32):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def safe_std(x, dim, eps=1e-6, keepdim=False):
    x = x.float()
    return torch.sqrt(torch.var(x, dim=dim, unbiased=False, keepdim=keepdim).clamp_min(0) + eps)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_choose_gn_groups(out_channels), out_channels)
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_choose_gn_groups(out_channels), out_channels),
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = self.gn2(self.conv2(x))
        if self.proj is not None:
            residual = self.proj(residual)
        return F.relu(x + residual, inplace=True)


class MAEResNetFeatureExtractor(nn.Module):
    """Torch MAE-ResNet activation extractor compatible with Drifting feature shapes."""

    def __init__(self, in_channels=3, base_channels=64, layers=(2, 2, 2, 2), freeze=True, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.freeze = freeze
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(base_channels), base_channels)
        channels = base_channels
        stages = []
        for stage_idx, blocks in enumerate(layers):
            out_channels = base_channels * (2 ** stage_idx)
            stride = 2 if stage_idx > 0 else 1
            stage = [BasicBlock(channels, out_channels, stride=stride)]
            for _ in range(1, blocks):
                stage.append(BasicBlock(out_channels, out_channels))
            stages.append(nn.Sequential(*stage))
            channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList([nn.GroupNorm(_choose_gn_groups(base_channels * (2 ** i)), base_channels * (2 ** i)) for i in range(4)])
        if checkpoint_path:
            payload = torch.load(checkpoint_path, map_location="cpu")
            state = payload.get("model", payload)
            self.load_state_dict(state, strict=False)
        if freeze:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(False if self.freeze else mode)
        return self

    def _process_feat(self, out, name, feat, patch_mean_size, patch_std_size, use_mean, use_std):
        b, c, h, w = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        out[name] = flat
        if use_mean:
            out[f"{name}_mean"] = flat.mean(dim=1, keepdim=True)
        if use_std:
            out[f"{name}_std"] = safe_std(flat, dim=1, keepdim=True)
        for size in patch_mean_size:
            if h % size == 0 and w % size == 0:
                grouped = feat.reshape(b, c, h // size, size, w // size, size).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, size * size, c)
                out[f"{name}_mean_{size}"] = grouped.mean(dim=2)
        for size in patch_std_size:
            if h % size == 0 and w % size == 0:
                grouped = feat.reshape(b, c, h // size, size, w // size, size).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, size * size, c)
                out[f"{name}_std_{size}"] = safe_std(grouped, dim=2)

    def forward(
        self,
        x,
        patch_mean_size: Optional[Iterable[int]] = (2, 4),
        patch_std_size: Optional[Iterable[int]] = (2, 4),
        use_mean=True,
        use_std=True,
        every_k_block=2,
    ) -> Dict[str, torch.Tensor]:
        out = {}
        out["norm_x"] = torch.sqrt((x.float() ** 2).mean(dim=(2, 3)) + 1e-6).unsqueeze(1)
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        self._process_feat(out, "conv1", x, patch_mean_size, patch_std_size, use_mean, use_std)
        for i, stage in enumerate(self.stages):
            block_outputs = []
            for block in stage:
                x = block(x)
                block_outputs.append(x)
            x = self.stage_norms[i](x)
            lname = f"layer{i + 1}"
            self._process_feat(out, lname, x, patch_mean_size, patch_std_size, use_mean, use_std)
            if every_k_block and every_k_block >= 1:
                for block_idx, feat in enumerate(block_outputs, start=1):
                    if block_idx % int(every_k_block) == 0:
                        self._process_feat(out, f"{lname}_blk{block_idx}", feat, patch_mean_size, patch_std_size, use_mean, use_std)
        return out
