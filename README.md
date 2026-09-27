# SegDrift

SegDrift is a PyTorch image-to-mask segmentation implementation built around
image-conditioned drifting, DINOv3 features, and optional CLS-prior
calibration.

The Python package, experiment configurations, and detailed usage guide live
in [`drifting_segmentation/`](drifting_segmentation/README.md).

## Installation

```bash
pip install -e drifting_segmentation
```

## Data and checkpoints

Datasets, training runs, generated predictions, and model checkpoints are not
stored in this repository. Update the dataset paths in the selected YAML file
under `drifting_segmentation/configs/` before training or evaluation.

## License

This project is released under the MIT License.
