import torch


def build_feature_input(image, mask_prob, mode="masked_image", alpha=0.45):
    mask_prob = mask_prob.clamp(0, 1)
    if mode == "mask_rgb":
        return mask_prob.repeat(1, 3, 1, 1) * 2 - 1
    if mode == "masked_image":
        return image * mask_prob
    if mode == "overlay":
        red = torch.zeros_like(image)
        red[:, 0] = 1.0
        return image * (1 - alpha * mask_prob) + red * (alpha * mask_prob)
    if mode == "concat_projected":
        # Deterministic 3-channel projection without learnable params.
        return torch.cat([image[:, :2], mask_prob * 2 - 1], dim=1)
    raise ValueError(f"Unknown feature input mode: {mode}")
