from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class ArrayMemoryBank:
    """Class-wise CPU ring buffer for masks/features."""

    def __init__(self, num_classes: int = 1, max_size: int = 1024, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int64)
        self.count = np.zeros(self.num_classes, dtype=np.int64)

    def _init_bank(self, sample_shape):
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros((self.num_classes, self.max_size, *self.feature_shape), dtype=self.dtype)

    def add(self, samples, labels=None):
        samples = samples.detach().cpu().numpy() if torch.is_tensor(samples) else np.asarray(samples)
        if labels is None:
            labels = np.zeros((samples.shape[0],), dtype=np.int64)
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        labels = labels.astype(np.int64) % self.num_classes
        if self.bank is None:
            self._init_bank(samples.shape[1:])
        for i in range(samples.shape[0]):
            lbl = int(labels[i])
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            self.count[lbl] = min(self.count[lbl] + 1, self.max_size)

    def sample(self, labels=None, n_samples: int = 1, device=None, fallback=None):
        if labels is None:
            if fallback is not None:
                labels = torch.zeros(fallback.shape[0], dtype=torch.long)
            else:
                labels = np.zeros((1,), dtype=np.int64)
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        labels_np = labels_np.astype(np.int64) % self.num_classes

        if self.bank is None or self.feature_shape is None:
            if fallback is None:
                raise RuntimeError("MemoryBank is empty and no fallback was provided")
            fb = fallback.detach()
            reps = [fb[i % fb.shape[0]].unsqueeze(0).repeat(n_samples, *([1] * (fb.ndim - 1))) for i in range(labels_np.shape[0])]
            return torch.stack(reps, dim=0).to(device or fb.device)

        out_indices = np.empty((labels_np.shape[0], n_samples), dtype=np.int64)
        for i, lbl in enumerate(labels_np):
            valid = int(self.count[int(lbl)])
            if valid <= 0:
                out_indices[i] = 0
            else:
                out_indices[i] = np.random.choice(valid, n_samples, replace=valid < n_samples)
        out = self.bank[labels_np[:, None], out_indices]
        return torch.from_numpy(out).to(device=device)

    def state_dict(self):
        return {
            "bank": self.bank,
            "feature_shape": self.feature_shape,
            "ptr": self.ptr,
            "count": self.count,
            "num_classes": self.num_classes,
            "max_size": self.max_size,
        }

    def load_state_dict(self, state):
        self.bank = state["bank"]
        self.feature_shape = tuple(state["feature_shape"]) if state["feature_shape"] is not None else None
        self.ptr = state["ptr"]
        self.count = state["count"]
        self.num_classes = int(state["num_classes"])
        self.max_size = int(state["max_size"])
