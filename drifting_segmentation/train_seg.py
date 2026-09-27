import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from drifting_segmentation.drift_loss import drift_loss
from drifting_segmentation.features.dino_cls_prior import DINOCLSCalibrationPrior
from drifting_segmentation.features.feature_builder import CombinedFeatureExtractor
from drifting_segmentation.datasets.segmentation_dataset import SegmentationPairDataset
from drifting_segmentation.losses.segmentation import segmentation_losses
from drifting_segmentation.memory_bank import ArrayMemoryBank
from drifting_segmentation.models.drift_unet import DriftUNet
from drifting_segmentation.models.image_conditioned_dit import ImageConditionedDriftDiT
from drifting_segmentation.utils.logger import TeeLogger
from drifting_segmentation.utils.metrics import binary_metrics
from drifting_segmentation.utils.checkpoint import load_checkpoint, save_checkpoint
from drifting_segmentation.utils.config import load_config
from drifting_segmentation.utils.device import resolve_device
from drifting_segmentation.utils.ema import ModelEMA
from drifting_segmentation.utils.lr import build_lr_lambda
from drifting_segmentation.utils.seg_vis import save_binary_overlay, save_heatmap


def save_drift_checkpoint(path, model, optimizer, scheduler, epoch, step, ema, config, cls_prior=None):
    save_checkpoint(path, model, optimizer, scheduler, epoch, step, ema, config)
    if cls_prior is not None:
        payload = torch.load(path, map_location="cpu")
        payload["prior"] = cls_prior.state_dict()
        torch.save(payload, path)

def build_model(config):
    cfg = config["model"]
    if cfg.get("generator_type", "dit") == "unet":
        return DriftUNet(
            image_channels=config["dataset"].get("image_channels", 3),
            noise_channels=cfg.get("noise_channels", 2),
            out_channels=cfg.get("out_channels", 1),
            base_channels=cfg.get("base_channels", 64),
            residual_prior=cfg.get("residual_prior", True),
            prior_channel_index=cfg.get("prior_channel_index", 1),
        )
    return ImageConditionedDriftDiT(
        input_size=cfg.get("input_size", 256),
        image_channels=config["dataset"].get("image_channels", 3),
        noise_channels=cfg.get("noise_channels", 1),
        out_channels=cfg.get("out_channels", 1),
        patch_size=cfg.get("patch_size", 16),
        hidden_size=cfg.get("hidden_size", 768),
        depth=cfg.get("depth", 12),
        num_heads=cfg.get("num_heads", 12),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        cond_dim=cfg.get("cond_dim", 768),
        n_cond_tokens=cfg.get("n_cond_tokens", 16),
        noise_classes=cfg.get("noise_classes", 64),
        noise_coords=cfg.get("noise_coords", 32),
        use_qknorm=cfg.get("use_qknorm", True),
        use_swiglu=cfg.get("use_swiglu", True),
        use_rope=cfg.get("use_rope", True),
        use_rmsnorm=cfg.get("use_rmsnorm", True),
        use_checkpoint=cfg.get("use_checkpoint", True),
        attn_fp32=cfg.get("attn_fp32", True),
    )


def build_cls_prior(config):
    pcfg = config.get("prior", {})
    if not pcfg.get("use_cls_calibration", False):
        return None
    return DINOCLSCalibrationPrior(
        model_name=pcfg.get("dinov3_name", config["feature"].get("dinov3_name", "vit_base_patch16_dinov3.lvd1689m")),
        pretrained=pcfg.get("dinov3_pretrained", True),
        image_size=pcfg.get("dinov3_image_size", config["feature"].get("dinov3_image_size", 224)),
        adapter_hidden=pcfg.get("cls_adapter_hidden", 1024),
        temperature=pcfg.get("cls_prior_temperature", 0.07),
        focal_alpha=pcfg.get("focal_alpha", 0.25),
        focal_gamma=pcfg.get("focal_gamma", 2.0),
        freeze_dino=pcfg.get("freeze_dino", True),
    )


def build_loader(config, split):
    dcfg = config["dataset"]
    tcfg = config["training"]
    dataset = SegmentationPairDataset(
        dcfg[f"{split}_image_dir"],
        dcfg[f"{split}_mask_dir"],
        image_size=dcfg.get("image_size", 256),
        mask_channels=1,
        num_mask_classes=1,
        mask_suffix=dcfg.get("mask_suffix", ""),
        split_file=dcfg.get(f"{split}_split_file"),
        augment=split == "train" and dcfg.get("augment", True),
        hflip_prob=dcfg.get("hflip_prob", 0.5),
        vflip_prob=dcfg.get("vflip_prob", 0.0),
        rot90_prob=dcfg.get("rot90_prob", 0.0),
        affine_prob=dcfg.get("affine_prob", 0.0),
        affine_rotate_deg=dcfg.get("affine_rotate_deg", 15.0),
        affine_scale_min=dcfg.get("affine_scale_min", 0.9),
        affine_scale_max=dcfg.get("affine_scale_max", 1.1),
        color_prob=dcfg.get("color_prob", 0.0),
        brightness=dcfg.get("brightness", 0.15),
        contrast=dcfg.get("contrast", 0.15),
        saturation=dcfg.get("saturation", 0.10),
    )
    return DataLoader(
        dataset,
        batch_size=tcfg.get("batch_size", 2) if split == "train" else config.get("validation", {}).get("batch_size", 2),
        shuffle=split == "train",
        num_workers=dcfg.get("num_workers", 4),
        pin_memory=dcfg.get("pin_memory", True),
        drop_last=split == "train",
    )


def sample_cfg(batch_size, config, device):
    fcfg = config.get("forward", {})
    cfg_min = fcfg.get("cfg_min", 1.0)
    cfg_max = fcfg.get("cfg_max", 4.0)
    neg_cfg_pw = fcfg.get("neg_cfg_pw", 5.0)
    no_cfg_frac = fcfg.get("no_cfg_frac", 0.0)
    frac = torch.rand(batch_size, device=device)
    pw = 1 - neg_cfg_pw
    if abs(pw) < 1e-6:
        cfg = torch.exp(torch.log(torch.tensor(cfg_min, device=device)) + frac * (torch.log(torch.tensor(cfg_max, device=device)) - torch.log(torch.tensor(cfg_min, device=device))))
    else:
        cfg = (cfg_min ** pw + frac * (cfg_max ** pw - cfg_min ** pw)) ** (1 / pw)
    if no_cfg_frac > 0:
        cfg = torch.where(torch.rand(batch_size, device=device) < no_cfg_frac, torch.ones_like(cfg), cfg)
    return cfg


def reshape_for_drift(feat, group):
    b_g, tokens, dim = feat.shape
    bsz = b_g // group
    return feat.reshape(bsz, group, tokens, dim).permute(0, 2, 1, 3).reshape(bsz * tokens, group, dim)


def compute_drift_loss(feature_encoder, image, pred_prob, gt_mask, neg_masks, config):
    fcfg = config["feature"]
    dcfg = config["drift"]
    gen_per = config.get("forward", {}).get("gen_per_image", 1)
    neg_per = config["memory"].get("neg_per_sample", 16)

    gen_input = feature_encoder.feature_input(image.repeat_interleave(gen_per, dim=0), pred_prob)
    pos_input = feature_encoder.feature_input(image, gt_mask)
    neg_image = image[:, None].repeat(1, neg_per, 1, 1, 1).reshape(-1, *image.shape[1:])
    neg_input = feature_encoder.feature_input(neg_image, neg_masks.reshape(-1, *gt_mask.shape[1:]))

    gen_features = feature_encoder(gen_input)
    with torch.no_grad():
        pos_features = feature_encoder(pos_input)
        neg_features = feature_encoder(neg_input)

    total = image.new_zeros(())
    info = {}
    for key, gen_feat in gen_features.items():
        pos_feat = pos_features[key]
        neg_feat = neg_features[key]
        gen_d = reshape_for_drift(gen_feat, gen_per)
        pos_d = pos_feat[:, None].repeat(1, 1, 1, 1).permute(0, 2, 1, 3).reshape(pos_feat.shape[0] * pos_feat.shape[1], 1, pos_feat.shape[2])
        neg_d = neg_feat.reshape(image.shape[0], neg_per, neg_feat.shape[1], neg_feat.shape[2]).permute(0, 2, 1, 3).reshape(image.shape[0] * neg_feat.shape[1], neg_per, neg_feat.shape[2])
        common = min(gen_d.shape[0], pos_d.shape[0], neg_d.shape[0])
        loss_k, info_k = drift_loss(gen_d[:common], pos_d[:common], neg_d[:common], R_list=dcfg.get("R_list", [0.2, 0.05, 0.02]))
        weight = feature_encoder.weight_for(key)
        total = total + weight * loss_k.mean()
        info[f"drift/{key}"] = loss_k.detach().mean()
        for ik, iv in info_k.items():
            info[f"drift/{key}/{ik}"] = iv
    return total, info


@torch.no_grad()
def validate(model, cls_prior, loader, config, device, workdir, step):
    model.eval()
    if cls_prior is not None:
        cls_prior.eval()
    totals = {}
    count = 0
    sample_dir = Path(workdir) / "samples" / f"step_{step:08d}"
    cfg_scale = config.get("sampling", {}).get("cfg_scale", 1.0)
    for batch in loader:
        image = batch["image"].to(device)
        mask = ((batch["mask"].to(device) + 1) * 0.5).clamp(0, 1)
        noise = torch.randn_like(mask)
        if cls_prior is not None:
            prior_prob = cls_prior(image)["prior_prob"]
            noise = torch.cat([noise, prior_prob], dim=1)
        logits = model(image, noise, cfg_scale=cfg_scale)
        pred = torch.sigmoid(logits) > config.get("sampling", {}).get("threshold", 0.5)
        metrics = binary_metrics(pred[:, 0], mask[:, 0] > 0.5)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        if count == 0:
            save_binary_overlay(sample_dir / "overlay.png", image[0], pred[0], mask[0])
            if cls_prior is not None:
                save_heatmap(sample_dir / "prior_heatmap.png", prior_prob[0], image[0])
        count += 1
    model.train()
    if cls_prior is not None:
        cls_prior.train()
    return {k: v / max(1, count) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(workdir / "log.txt")
    device = resolve_device(args.device, config)
    logger.log(f"Using device: {device}")
    logger.log(f"Config: {config}")

    train_loader = build_loader(config, "train")
    val_loader = build_loader(config, "val")
    model = build_model(config).to(device)
    cls_prior = build_cls_prior(config)
    if cls_prior is not None:
        cls_prior = cls_prior.to(device)
    feature_encoder = CombinedFeatureExtractor(config["feature"]).to(device).eval()
    ema = ModelEMA(model, decay=config["training"].get("ema_decay", 0.999)).to(device)
    train_params = list(model.parameters())
    if cls_prior is not None:
        train_params += [p for p in cls_prior.parameters() if p.requires_grad]
    optimizer = AdamW(
        train_params,
        lr=config["training"].get("learning_rate", 5e-5),
        betas=(config["training"].get("adam_b1", 0.9), config["training"].get("adam_b2", 0.95)),
        weight_decay=config["training"].get("weight_decay", 0.01),
    )
    total_steps = config["training"].get("total_steps", 100000)
    scheduler = LambdaLR(optimizer, build_lr_lambda(total_steps, config["training"].get("warmup_steps", 10000), config["training"].get("lr_schedule", "constant")))
    pos_bank = ArrayMemoryBank(1, config["memory"].get("positive_bank_size", 128))
    neg_bank = ArrayMemoryBank(1, config["memory"].get("negative_bank_size", 2048))

    step = 0
    best_metric = float("-inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, ema, map_location=device)
        if cls_prior is not None and ckpt.get("prior") is not None:
            cls_prior.load_state_dict(ckpt["prior"], strict=True)
        step = ckpt.get("step", 0)
        best_metric = ckpt.get("best_metric", best_metric)

    scaler = torch.cuda.amp.GradScaler(enabled=config["training"].get("amp", True) and device.type == "cuda")
    train_iter = iter(train_loader)
    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        image = batch["image"].to(device, non_blocking=True)
        mask = ((batch["mask"].to(device, non_blocking=True) + 1) * 0.5).clamp(0, 1)
        labels = torch.zeros(image.shape[0], dtype=torch.long)
        pos_bank.add(mask, labels)
        # C3: prediction-aware negative mining.
        # negative_source controls what goes into neg_bank:
        #   "gt"         - original behaviour: store GT mask
        #   "prediction" - store model false-positive regions as hard negatives
        #   "mixed"      - blend both according to historical_pred_ratio (default 0.5)
        neg_source = config["memory"].get("negative_source", "gt")
        warmup_done = step >= config["memory"].get("warmup_steps", 500)
        if neg_source == "gt" or not warmup_done:
            neg_bank.add(mask, labels)
        elif neg_source == "mixed":
            neg_bank.add(mask, labels)  # keep GT neg stream as well
        # hard negatives from prediction added AFTER forward pass (see below)
        neg_masks = neg_bank.sample(labels, config["memory"].get("neg_per_sample", 16), device=device, fallback=mask)

        gen_per = config.get("forward", {}).get("gen_per_image", 1)
        image_gen = image.repeat_interleave(gen_per, dim=0)
        noise = torch.randn(image.shape[0] * gen_per, 1, image.shape[-2], image.shape[-1], device=device)
        prior_loss = image.new_zeros(())
        prior_logs = {}
        if cls_prior is not None:
            prior_out = cls_prior(image, gt_mask=mask)
            prior_prob = prior_out["prior_prob"]
            prior_input = prior_prob.detach() if config.get("prior", {}).get("detach_prior_input", True) else prior_prob
            noise = torch.cat([noise, prior_input.repeat_interleave(gen_per, dim=0)], dim=1)
            prior_loss = prior_out["loss_cls_prior"]
            prior_logs = {
                "loss_cls_prior": prior_out["loss_cls_prior"].detach(),
                "loss_cls_focal": prior_out["loss_cls_focal"],
                "loss_cls_dice": prior_out["loss_cls_dice"],
                "prior_mean": prior_prob.detach().mean(),
                "prior_min": prior_prob.detach().amin(),
                "prior_max": prior_prob.detach().amax(),
            }
        cfg_scale = sample_cfg(image.shape[0], config, device).repeat_interleave(gen_per)

        optimizer.zero_grad(set_to_none=True)
        if cls_prior is not None:
            for param in cls_prior.parameters():
                if param.requires_grad:
                    param.grad = None
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(image_gen, noise, cfg_scale=cfg_scale)
            prob = torch.sigmoid(logits)
            drift, drift_info = compute_drift_loss(feature_encoder, image, prob, mask, neg_masks, config)
            first_logits = logits.reshape(image.shape[0], gen_per, *logits.shape[1:])[:, 0]
            seg, seg_info = segmentation_losses(
                first_logits,
                mask,
                lambda_bce=config["loss"].get("lambda_bce", 0.5),
                lambda_dice=config["loss"].get("lambda_dice", 0.5),
                lambda_boundary=config["loss"].get("lambda_boundary", 0.05),
                lambda_lovasz=config["loss"].get("lambda_lovasz", 0.0),
            )
            loss = (
                config["loss"].get("lambda_drift", 0.1) * drift
                + seg
                + config.get("prior", {}).get("lambda_cls_prior", 0.0) * prior_loss
            )
        if not torch.isfinite(loss):
            logger.log(f"step={step} non-finite loss, skipping")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        params_for_clip = list(model.parameters())
        if cls_prior is not None:
            params_for_clip += [p for p in cls_prior.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, config["training"].get("grad_clip", 1.0))
        if not torch.isfinite(grad_norm):
            logger.log(f"step={step} non-finite grad, skipping")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        # C3: add hard negatives (false-positive regions) to neg_bank after the step,
        # so the bank is populated with regions the current model is actively confused about.
        if neg_source in ("prediction", "mixed") and warmup_done:
            with torch.no_grad():
                pred_prob_first = prob.reshape(
                    image.shape[0], gen_per, *prob.shape[1:]
                )[:, 0].detach()                                      # [B, 1, H, W]
                hard_neg_thresh = config["memory"].get("hard_neg_thresh", 0.3)
                hard_neg = (pred_prob_first > hard_neg_thresh) & (mask < 0.5)
                hard_neg_mask = hard_neg.float()
                # only add when there are actual false-positive pixels to avoid
                # storing all-zero masks that carry no signal
                valid = (hard_neg_mask.flatten(1).sum(1) > 0).cpu()   # [B] on cpu
                if valid.any():
                    neg_bank.add(hard_neg_mask.cpu()[valid], labels[valid])

        step += 1

        if step % config["training"].get("log_every", 20) == 0:
            logger.log(
                f"step={step} lr={scheduler.get_last_lr()[0]:.6g} loss={float(loss.detach()):.4f} "
                f"drift={float(drift.detach()):.4f} seg={float(seg.detach()):.4f} "
                f"bce={float(seg_info['loss_bce']):.4f} dice={float(seg_info['loss_dice']):.4f} "
                f"lovasz={float(seg_info['loss_lovasz']):.4f} "
                f"cls_prior={float(prior_loss.detach()):.4f} "
                f"prior_mean={float(prior_logs.get('prior_mean', torch.tensor(0.0))):.4f} "
                f"prior_min={float(prior_logs.get('prior_min', torch.tensor(0.0))):.4f} "
                f"prior_max={float(prior_logs.get('prior_max', torch.tensor(0.0))):.4f} "
                f"g_norm={float(grad_norm):.4f} cfg={float(cfg_scale.mean()):.4f}"
            )
        if step % config["training"].get("val_every", 500) == 0:
            metrics = validate(ema.module, cls_prior, val_loader, config, device, workdir, step)
            logger.log("validation " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            current_metric = metrics.get(config["training"].get("best_metric", "dice"), float("-inf"))
            if current_metric > best_metric:
                best_metric = current_metric
                save_drift_checkpoint(workdir / "best.pt", model, optimizer, scheduler, 0, step, ema, config, cls_prior)
                best_payload = torch.load(workdir / "best.pt", map_location="cpu")
                best_payload["best_metric"] = best_metric
                torch.save(best_payload, workdir / "best.pt")
                logger.log(f"saved best.pt {config['training'].get('best_metric', 'dice')}={best_metric:.4f}")
        if step % config["training"].get("ckpt_every", 1000) == 0:
            save_drift_checkpoint(workdir / "last.pt", model, optimizer, scheduler, 0, step, ema, config, cls_prior)
            last_payload = torch.load(workdir / "last.pt", map_location="cpu")
            last_payload["best_metric"] = best_metric
            torch.save(last_payload, workdir / "last.pt")
    save_drift_checkpoint(workdir / "last.pt", model, optimizer, scheduler, 0, step, ema, config, cls_prior)
    last_payload = torch.load(workdir / "last.pt", map_location="cpu")
    last_payload["best_metric"] = best_metric
    torch.save(last_payload, workdir / "last.pt")


if __name__ == "__main__":
    main()
