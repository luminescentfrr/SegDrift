import math

import cv2
import numpy as np
import torch


def _as_batch(mask):
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    return mask


def _safe_ratio(num, denom, empty_value):
    if denom <= 0:
        return float(empty_value)
    return float(num / denom)


def _boundary(mask_np):
    mask_np = mask_np.astype(np.uint8)
    if mask_np.sum() == 0:
        return mask_np.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    return (mask_np > 0) & (eroded == 0)


def _hd95_single(pred_np, target_np):
    pred_np = pred_np.astype(bool)
    target_np = target_np.astype(bool)
    pred_empty = not pred_np.any()
    target_empty = not target_np.any()
    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        height, width = pred_np.shape[-2:]
        return float(math.sqrt(height * height + width * width))

    pred_boundary = _boundary(pred_np)
    target_boundary = _boundary(target_np)
    if not pred_boundary.any() or not target_boundary.any():
        height, width = pred_np.shape[-2:]
        return float(math.sqrt(height * height + width * width))

    dist_to_target = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([dist_to_target[pred_boundary], dist_to_pred[target_boundary]])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def binary_metrics(pred, target):
    pred = _as_batch(pred).bool()
    target = _as_batch(target).bool()
    pred_cpu = pred.detach().cpu()
    target_cpu = target.detach().cpu()

    scores = {"dice": [], "iou": [], "miou": [], "recall": [], "hd95": []}
    for pred_i, target_i in zip(pred_cpu, target_cpu):
        pred_i = pred_i.bool()
        target_i = target_i.bool()
        tp = (pred_i & target_i).sum().item()
        fp = (pred_i & ~target_i).sum().item()
        fn = (~pred_i & target_i).sum().item()

        pred_pos = tp + fp
        target_pos = tp + fn
        fg_union = tp + fp + fn

        dice = _safe_ratio(2 * tp, pred_pos + target_pos, empty_value=1.0 if pred_pos == 0 else 0.0)
        iou = _safe_ratio(tp, fg_union, empty_value=1.0)
        recall = _safe_ratio(tp, target_pos, empty_value=1.0 if pred_pos == 0 else 0.0)

        scores["dice"].append(dice)
        scores["iou"].append(iou)
        scores["miou"].append(iou)
        scores["recall"].append(recall)
        scores["hd95"].append(_hd95_single(pred_i.numpy(), target_i.numpy()))

    return {key: float(np.mean(value)) for key, value in scores.items()}
