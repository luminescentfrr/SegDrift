import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from drifting_segmentation.train_seg import build_cls_prior, build_model
from drifting_segmentation.utils.checkpoint import load_checkpoint
from drifting_segmentation.utils.config import load_config
from drifting_segmentation.utils.device import resolve_device
from drifting_segmentation.utils.seg_vis import save_binary_overlay, save_heatmap, save_mask


def load_image(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy((image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device, config)
    model = build_model(config).to(device)
    cls_prior = build_cls_prior(config)
    if cls_prior is not None:
        cls_prior = cls_prior.to(device)
    ckpt = load_checkpoint(args.ckpt_path, model, map_location=device)
    if args.use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=True)
    if cls_prior is not None and ckpt.get("prior") is not None:
        cls_prior.load_state_dict(ckpt["prior"], strict=True)
    model.eval()
    if cls_prior is not None:
        cls_prior.eval()

    input_path = Path(args.input)
    out_path = Path(args.output)
    paths = sorted(input_path.iterdir()) if input_path.is_dir() else [input_path]
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else config.get("sampling", {}).get("cfg_scale", 1.0)
    threshold = config.get("sampling", {}).get("threshold", 0.5)
    for path in paths:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        image = load_image(path, config["dataset"].get("image_size", 256)).unsqueeze(0).to(device)
        noise = torch.randn(image.shape[0], 1, image.shape[-2], image.shape[-1], device=device)
        if cls_prior is not None:
            prior_prob = cls_prior(image)["prior_prob"]
            noise = torch.cat([noise, prior_prob], dim=1)
        logits = model(image, noise, cfg_scale=cfg_scale)
        pred = torch.sigmoid(logits) > threshold
        target = out_path / f"{path.stem}_mask.png" if input_path.is_dir() else out_path
        save_mask(target, pred[0])
        save_binary_overlay(target.with_name(f"{target.stem}_overlay.png"), image[0], pred[0])
        if cls_prior is not None:
            save_heatmap(target.with_name(f"{target.stem}_prior_heatmap.png"), prior_prob[0], image[0])


if __name__ == "__main__":
    main()
