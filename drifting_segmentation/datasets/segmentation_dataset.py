from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class SegmentationPairDataset(Dataset):
    """Paired image/mask dataset with configurable train-time augmentation."""

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=256,
        mask_channels=1,
        num_mask_classes=1,
        mask_suffix="",
        split_file=None,
        augment=False,
        hflip_prob=0.5,
        vflip_prob=0.0,
        rot90_prob=0.0,
        affine_prob=0.0,
        affine_rotate_deg=15.0,
        affine_scale_min=0.9,
        affine_scale_max=1.1,
        color_prob=0.0,
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.mask_channels = mask_channels
        self.num_mask_classes = num_mask_classes
        self.mask_suffix = mask_suffix
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.affine_prob = affine_prob
        self.affine_rotate_deg = affine_rotate_deg
        self.affine_scale_min = affine_scale_min
        self.affine_scale_max = affine_scale_max
        self.color_prob = color_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

        if split_file:
            stems = [line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()]
            image_paths = [self._find_by_stem(self.image_dir, stem) for stem in stems]
        else:
            image_paths = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

        self.pairs = []
        for image_path in image_paths:
            mask_path = self._find_by_stem(self.mask_dir, image_path.stem + self.mask_suffix)
            if mask_path is not None:
                self.pairs.append((image_path, mask_path))
        if not self.pairs:
            example = image_paths[0].stem + self.mask_suffix if image_paths else "<image_stem>"
            raise RuntimeError(
                f"No image/mask pairs found in {self.image_dir} and {self.mask_dir}. "
                f"Expected masks like {example}.png; set dataset.mask_suffix if needed."
            )

    @staticmethod
    def _find_by_stem(root, stem):
        root = Path(root)
        for ext in IMAGE_EXTS:
            path = root / f"{stem}{ext}"
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return image

    def _load_mask(self, path):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return mask

    def _encode_mask(self, mask):
        if self.mask_channels == 1:
            mask = (mask > 0).astype(np.float32)
            mask = mask[None] * 2.0 - 1.0
            return torch.from_numpy(mask)

        mask_id = np.clip(mask.astype(np.int64), 0, self.num_mask_classes - 1)
        onehot = np.eye(self.num_mask_classes, dtype=np.float32)[mask_id]
        onehot = onehot.transpose(2, 0, 1)
        return torch.from_numpy(onehot * 2.0 - 1.0)

    def _apply_geometric_augment(self, image, mask):
        if np.random.rand() < self.hflip_prob:
            image = image[:, ::-1]
            mask = mask[:, ::-1]

        if np.random.rand() < self.vflip_prob:
            image = image[::-1]
            mask = mask[::-1]

        if np.random.rand() < self.rot90_prob:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

        if np.random.rand() < self.affine_prob:
            h, w = image.shape[:2]
            angle = np.random.uniform(-self.affine_rotate_deg, self.affine_rotate_deg)
            scale = np.random.uniform(self.affine_scale_min, self.affine_scale_max)
            matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
            image = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def _apply_color_augment(self, image):
        if np.random.rand() >= self.color_prob:
            return image

        image_f = image.astype(np.float32)
        if self.contrast > 0:
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            image_f = image_f * factor
        if self.brightness > 0:
            offset = 255.0 * np.random.uniform(-self.brightness, self.brightness)
            image_f = image_f + offset
        image = np.clip(image_f, 0, 255).astype(np.uint8)

        if self.saturation > 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        if self.augment:
            image, mask = self._apply_geometric_augment(image, mask)
            image = self._apply_color_augment(image)

        image = image.astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = self._encode_mask(mask)
        return {
            "image": image,
            "mask": mask,
            "name": image_path.stem,
        }
