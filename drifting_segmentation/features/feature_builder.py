import torch
import torch.nn as nn

from .convnext_features import ConvNeXtFeatureExtractor
from .dinov3_features import DINOv3FeatureExtractor
from .feature_inputs import build_feature_input
from .mae_resnet import MAEResNetFeatureExtractor


class CombinedFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_mode = config.get("input_mode", "masked_image")
        self.selected = config.get("selected", {})
        self.weights = config.get("weights", {})
        self.use_convnext = config.get("use_convnext", True)
        self.use_mae = config.get("use_mae", True)
        self.use_dinov3 = config.get("use_dinov3", False)
        if self.use_dinov3:
            self.dinov3 = DINOv3FeatureExtractor(
                model_name=config.get("dinov3_name", "vit_base_patch16_dinov3.lvd1689m"),
                pretrained=config.get("dinov3_pretrained", True),
                freeze=config.get("freeze", True),
                image_size=config.get("dinov3_image_size", 224),
            )
        else:
            self.dinov3 = None
        if self.use_convnext:
            self.convnext = ConvNeXtFeatureExtractor(
                model_name=config.get("convnext_name", "convnextv2_base.fcmae_ft_in22k_in1k"),
                pretrained=config.get("convnext_pretrained", True),
                freeze=config.get("freeze", True),
            )
        else:
            self.convnext = None
        if self.use_mae:
            self.mae = MAEResNetFeatureExtractor(
                checkpoint_path=config.get("mae_path") or None,
                freeze=config.get("freeze", True),
            )
        else:
            self.mae = None

    def feature_input(self, image, mask_prob):
        if self.input_mode == "feature_weighted":
            # C2: return the raw image — the mask weighting happens inside forward_weighted.
            # We still need a pixel-space tensor to pass through the generic pipeline,
            # so return image as-is and carry mask_prob alongside via a 2-tuple.
            return (image, mask_prob)
        return build_feature_input(image, mask_prob, mode=self.input_mode)

    def forward(self, x):
        # C2: feature_weighted mode passes (image, mask_prob) as a tuple instead of a
        # pre-masked pixel tensor.
        if isinstance(x, tuple):
            image, mask_prob = x
            out = {}
            if self.dinov3 is not None:
                for key, value in self.dinov3.forward_weighted(image, mask_prob).items():
                    out[f"dinov3/{key}"] = value
            if not self.selected:
                return out
            selected_keys = []
            if isinstance(self.selected, dict):
                for prefix, keys in self.selected.items():
                    selected_keys.extend([f"{prefix}/{key}" for key in keys])
            else:
                selected_keys = list(self.selected)
            return {key: out[key] for key in selected_keys if key in out}

        out = {}
        if self.dinov3 is not None:
            for key, value in self.dinov3(x).items():
                out[f"dinov3/{key}"] = value
        if self.convnext is not None:
            for key, value in self.convnext(x).items():
                out[f"convnext/{key}"] = value
        if self.mae is not None:
            for key, value in self.mae(x).items():
                out[f"mae/{key}"] = value
        if not self.selected:
            return out
        selected_keys = []
        if isinstance(self.selected, dict):
            for prefix, keys in self.selected.items():
                selected_keys.extend([f"{prefix}/{key}" for key in keys])
        else:
            selected_keys = list(self.selected)
        return {key: out[key] for key in selected_keys if key in out}

    def weight_for(self, key):
        return float(self.weights.get(key, self.weights.get(key.split("/", 1)[-1], 1.0)))
