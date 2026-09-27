# Drifting Segmentation

Self-contained Torch implementation of a Drifting-style image-to-mask segmentation model.

The current default research path is:

```text
image + noise + DINOv3 CLS prior -> Image-conditioned LightningDiT -> mask
```

An alternative 5-level U-Net generator is also included:

```text
image + noise + DINOv3 CLS prior -> 5-level residual U-Net -> mask
```

It supports:

- DINOv3 feature drift loss
- optional DINOv3 CLS calibration prior
- BCE / Dice / Boundary / Lovasz losses
- EMA, resume, best/last checkpoints
- heatmap visualization for CLS prior

## Install

From this folder:

```bash
pip install -e .
```

Or from the repository root:

```bash
pip install -e drifting_segmentation
```

## Data Layout

Images and masks are paired by filename stem. For ISIC-style masks:

```text
images/ISIC_0000000.jpg
masks/ISIC_0000000_segmentation.png
```

set:

```yaml
dataset:
  mask_suffix: _segmentation
  augment: true
  hflip_prob: 0.5
  vflip_prob: 0.5
  rot90_prob: 0.5
  affine_prob: 0.5
  affine_rotate_deg: 15.0
  affine_scale_min: 0.9
  affine_scale_max: 1.1
  color_prob: 0.5
  brightness: 0.12
  contrast: 0.12
  saturation: 0.08
```

## Configs

Baseline DINOv3 drifting:

```text
drifting_segmentation/configs/isic_binary_256_full.yaml
```

DINOv3 CLS calibration prior + Lovasz:

```text
drifting_segmentation/configs/isic_binary_256_cls_prior.yaml
```

5-level U-Net generator + DINOv3 CLS calibration prior + Lovasz:

```text
drifting_segmentation/configs/isic_binary_256_cls_prior_unet.yaml
```

Important fields:

```yaml
model:
  generator_type: dit  # use unet for the 5-level U-Net config
  noise_channels: 2

loss:
  lambda_drift: 0.1
  lambda_bce: 0.5
  lambda_dice: 0.8
  lambda_boundary: 0.1
  lambda_lovasz: 0.3

prior:
  use_cls_calibration: true
  lambda_cls_prior: 0.1
  detach_prior_input: true

training:
  best_metric: dice
```

## Train

```bash
python -m drifting_segmentation.train_seg \
  --config drifting_segmentation/configs/ablation_c1.yaml \
  --workdir runs/ISIC17/ablation_c1 \
  --device cuda:0
```

Only two checkpoints are maintained:

```text
runs/drift_isic_cls_prior/last.pt
runs/drift_isic_cls_prior/best.pt
```

Logs are written to:

```text
runs/drift_isic_cls_prior/log.txt
```

U-Net variant:

```bash
python -m drifting_segmentation.train_seg \
  --config drifting_segmentation/configs/ablation_c1c2.yaml \
  --workdir runs/Kvasir-SEG/ablation_c1c2 \
```

## Resume

```bash
python -m drifting_segmentation.train_seg \
  --config drifting_segmentation/configs/isic_binary_256_cls_prior.yaml \
  --workdir runs/drift_isic_cls_prior \
  --device cuda:0 \
  --resume runs/drift_isic_cls_prior/last.pt
```

Resume requires the same model shape. For example, baseline uses `noise_channels: 1`, CLS-prior uses `noise_channels: 2`, and the U-Net config uses a different generator from DiT, so these checkpoints are not directly interchangeable.

## Evaluate

```bash
python -m drifting_segmentation.evaluate_seg \
  --config drifting_segmentation/configs/baseline_kvasir.yaml \
  --ckpt-path runs/Kvasir-SEG/baseline/best.pt \
  --workdir runs/Kvasir-SEG/baseline/eval \
  --device cuda:0 \
  --use-ema
```

The evaluator reports:

```text
dice
iou
pred_fg_ratio
gt_fg_ratio
```

## Infer

Single image:

```bash
python -m drifting_segmentation.infer_seg \
  --config drifting_segmentation/configs/isic_binary_256_cls_prior.yaml \
  --ckpt-path runs/ISIC17/FULL/best.pt \
  --input path/to/image.jpg \
  --output predictions/image_mask.png \
  --device cuda:0 \
  --use-ema
```

Folder:

```bash
python -m drifting_segmentation.infer_seg \
  --config drifting_segmentation/configs/full_kvasir.yaml \
  --ckpt-path runs/Kvasir-SEG/FULL/best.pt \
  --input /mnt/c/tianyu_project/datasets/Kvasir-SEG/val/images \
  --output runs/Kvasir-SEG/FULL/predictions \
  --device cuda:0 \
  --use-ema
```

Outputs include:

```text
*_mask.png
*_overlay.png
*_prior_heatmap.png  # when CLS prior is enabled
```

## Shell Note

When using multi-line shell commands, the backslash must be the final character on the line:

```bash
python -m drifting_segmentation.evaluate_seg \
  --config drifting_segmentation/configs/isic_binary_256_cls_prior.yaml \
  --ckpt-path runs/drift_isic_cls_prior/best.pt \
  --workdir runs/drift_isic_cls_prior/eval
```

Do not add spaces after `\`.
