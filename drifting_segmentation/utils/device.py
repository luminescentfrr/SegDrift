import torch


def resolve_device(device_arg=None, config=None):
    runtime_cfg = (config or {}).get("runtime", {})
    requested = device_arg or runtime_cfg.get("device", "auto")
    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
        torch.cuda.set_device(device)
    return device
