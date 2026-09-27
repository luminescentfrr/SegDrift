# SegDrift: Prior-Guided Semantic Drifting with Hard Negative Mining for Medical Image Segmentation

## Installation

```bash
git clone https://github.com/luminescentfrr/SegDrift.git
cd SegDrift
conda create -n segdrift python=3.10
conda activate segdrift
pip install -e drifting_segmentation
```

## Usage

Update the dataset paths in the selected file under
`drifting_segmentation/configs/`, then run the required command.

Train:

```bash
python -m drifting_segmentation.train_seg \
  --config drifting_segmentation/configs/full_isic.yaml \
  --workdir runs/ISIC17/FULL \
  --device cuda:0
```

Evaluate:

```bash
python -m drifting_segmentation.evaluate_seg \
  --config drifting_segmentation/configs/full_isic.yaml \
  --ckpt-path runs/ISIC17/FULL/best.pt \
  --workdir runs/ISIC17/FULL/eval \
  --device cuda:0 \
  --use-ema
```

Inference:

```bash
python -m drifting_segmentation.infer_seg \
  --config drifting_segmentation/configs/full_isic.yaml \
  --ckpt-path runs/ISIC17/FULL/best.pt \
  --input path/to/image.jpg \
  --output predictions/image_mask.png \
  --device cuda:0 \
  --use-ema
```
