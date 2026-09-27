import torch
import torch.nn.functional as F


def dice_loss(prob, target, eps=1e-6):
    prob = prob.float()
    target = target.float()
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def boundary_loss(prob, target):
    sobel_x = prob.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3)
    sobel_y = prob.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).reshape(1, 1, 3, 3)
    pred_edge = torch.sqrt(F.conv2d(prob, sobel_x, padding=1).pow(2) + F.conv2d(prob, sobel_y, padding=1).pow(2) + 1e-6)
    gt_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1).pow(2) + F.conv2d(target, sobel_y, padding=1).pow(2) + 1e-6)
    return F.l1_loss(pred_edge, gt_edge)


def lovasz_grad(gt_sorted):
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    labels = labels.reshape(-1)
    logits = logits.reshape(-1)
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge_loss(logits, target, per_image=True):
    target = (target > 0.5).float()
    if per_image:
        losses = [lovasz_hinge_flat(logit, label) for logit, label in zip(logits, target)]
        return torch.stack(losses).mean()
    return lovasz_hinge_flat(logits, target)


def segmentation_losses(
    logits,
    target,
    lambda_bce=0.5,
    lambda_dice=0.5,
    lambda_boundary=0.05,
    lambda_lovasz=0.0,
):
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    dice = dice_loss(prob, target)
    boundary = boundary_loss(prob, target) if lambda_boundary > 0 else logits.new_zeros(())
    lovasz = lovasz_hinge_loss(logits, target) if lambda_lovasz > 0 else logits.new_zeros(())
    total = lambda_bce * bce + lambda_dice * dice + lambda_boundary * boundary + lambda_lovasz * lovasz
    return total, {
        "loss_bce": bce.detach(),
        "loss_dice": dice.detach(),
        "loss_boundary": boundary.detach(),
        "loss_lovasz": lovasz.detach(),
    }
