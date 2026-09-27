import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Stable pairwise distance for [B, N, D] and [B, M, D]."""
    x = x.float()
    y = y.float()
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Iterable[float] = (0.2, 0.05, 0.02),
) -> Tuple[torch.Tensor, dict]:
    """Torch implementation of Drifting loss.

    Args:
        gen: generated features, [B, Cg, S].
        fixed_pos: positive detached features, [B, Cp, S].
        fixed_neg: negative detached features, [B, Cn, S].
    Returns:
        Per-batch loss [B] and scalar info dict.
    """
    gen = gen.float()
    fixed_pos = fixed_pos.float().detach()
    if fixed_neg is None:
        fixed_neg = gen[:, :0, :].detach()
    else:
        fixed_neg = fixed_neg.float().detach()

    bsz, c_g, feat_dim = gen.shape
    c_n = fixed_neg.shape[1]
    c_p = fixed_pos.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        dist = cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / torch.clamp(targets_w.mean(), min=1e-8)
        scale_inputs = torch.clamp(scale / math.sqrt(feat_dim), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / torch.clamp(scale, min=1e-3)

        diag_mask = torch.eye(c_g, device=gen.device, dtype=torch.float32)
        block_mask = F.pad(diag_mask, (0, c_n + c_p)).unsqueeze(0)
        dist_normed = dist_normed + block_mask * 100.0

        force_across_R = torch.zeros_like(old_gen_scaled)
        info = {"scale": scale.detach()}
        for R in R_list:
            logits = -dist_normed / float(R)
            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
            affinity = affinity * targets_w[:, None, :]

            split_idx = c_g + c_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg
            r_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force_R = torch.einsum("biy,byx->bix", r_coeff, targets_scaled)
            total_coeffs = r_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = (total_force_R ** 2).mean()
            info[f"loss_{R}"] = f_norm_val.detach()
            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = (old_gen_scaled + force_across_R).detach()

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))
    return loss, {key: value.detach().mean() for key, value in info.items()}
