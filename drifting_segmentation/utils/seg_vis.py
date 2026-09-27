from pathlib import Path

import cv2
import numpy as np


def tensor_image_to_uint8(image):
    image = image.detach().cpu().float()
    image = ((image + 1) * 127.5).clamp(0, 255).byte()
    return image.permute(1, 2, 0).numpy()


def save_binary_overlay(path, image, pred_mask, gt_mask=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor_image_to_uint8(image)
    pred = pred_mask.detach().cpu().squeeze().numpy().astype(bool)
    overlay = img.copy()
    overlay[pred] = (0.55 * overlay[pred] + np.array([255, 40, 40]) * 0.45).astype(np.uint8)
    if gt_mask is not None:
        gt = gt_mask.detach().cpu().squeeze().numpy()
        gt = gt > 0 if gt.dtype != bool else gt
        border = gt & ~pred
        overlay[border] = (40, 220, 80)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = mask.detach().cpu().squeeze().numpy()
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), arr)


def save_heatmap(path, values, image=None, alpha=0.45):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = values.detach().cpu().float().squeeze().numpy()
    arr = arr - arr.min()
    denom = arr.max()
    if denom > 1e-8:
        arr = arr / denom
    heat = cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if image is not None:
        base = tensor_image_to_uint8(image)
        base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        heat = cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0)
    cv2.imwrite(str(path), heat)
