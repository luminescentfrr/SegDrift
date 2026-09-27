import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from drifting_segmentation.train_seg import build_cls_prior, build_model
from drifting_segmentation.utils.metrics import binary_metrics
from drifting_segmentation.datasets.segmentation_dataset import SegmentationPairDataset
from drifting_segmentation.utils.checkpoint import load_checkpoint
from drifting_segmentation.utils.config import load_config
from drifting_segmentation.utils.device import resolve_device
from drifting_segmentation.utils.seg_vis import save_binary_overlay


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--workdir", required=True)
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

    dcfg = config["dataset"]
    dataset = SegmentationPairDataset(
        dcfg["val_image_dir"],
        dcfg["val_mask_dir"],
        image_size=dcfg.get("image_size", 256),
        mask_channels=1,
        mask_suffix=dcfg.get("mask_suffix", ""),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=config.get("validation", {}).get("batch_size", 2), shuffle=False)
    out_dir = Path(args.workdir)
    totals = {}
    count = 0
    infer_time = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        mask = ((batch["mask"].to(device) + 1) * 0.5).clamp(0, 1)
        noise = torch.randn_like(mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if cls_prior is not None:
            prior_prob = cls_prior(image)["prior_prob"]
            noise = torch.cat([noise, prior_prob], dim=1)
        logits = model(image, noise, cfg_scale=config.get("sampling", {}).get("cfg_scale", 1.0))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_time += time.perf_counter() - start
        pred = torch.sigmoid(logits) > config.get("sampling", {}).get("threshold", 0.5)
        metrics = binary_metrics(pred[:, 0], mask[:, 0] > 0.5)
        batch_size = image.shape[0]
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v * batch_size
        for i, name in enumerate(batch["name"]):
            save_binary_overlay(out_dir / "overlays" / f"{name}.png", image[i], pred[i], mask[i])
        count += batch_size
    results = {k: v / max(1, count) for k, v in totals.items()}
    results["fps"] = count / infer_time if infer_time > 0 else 0.0
    print(results)


if __name__ == "__main__":
    main()
