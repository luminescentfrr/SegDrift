import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from drifting_segmentation.features.dinov3_features import DINOv3FeatureExtractor
except ModuleNotFoundError:
    sys.path.append(os.fspath(Path(__file__).resolve().parents[1]))
    from drifting_segmentation.features.dinov3_features import DINOv3FeatureExtractor


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VISUAL_SUFFIXES = (
    "_mask_prior_heatmap",
    "_prior_heatmap",
    "_mask_overlay",
    "_overlay",
    "_heatmap",
    "_img",
    "_image",
    "_gt",
)
PRED_SUFFIXES = ("_pred", "_mask", "_segmentation")


def parse_pred_arg(value):
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def canonical_prediction_stem(path):
    stem = Path(path).stem
    for suffix in VISUAL_SUFFIXES + PRED_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def numbered_prediction_index(path):
    match = re.match(r"^(?:test|val|sample)_(\d+)_(?:pred|mask|segmentation)$", Path(path).stem)
    if match:
        return int(match.group(1))
    return None


def prediction_candidate_rank(path):
    path = Path(path)
    stem = path.stem
    lower_name = path.name.lower()
    if any(stem.endswith(suffix) for suffix in VISUAL_SUFFIXES):
        return None
    if "overlay" in lower_name or "heatmap" in lower_name:
        return None

    rank = 100
    parent_names = {p.name.lower() for p in path.parents}
    if "pred_masks" in parent_names or "preds" in parent_names or "predictions" in parent_names:
        rank -= 30
    if stem.endswith("_pred"):
        rank -= 25
    elif stem.endswith("_mask"):
        rank -= 20
    elif stem.endswith("_segmentation"):
        rank -= 15
    else:
        rank -= 5
    return rank


def build_prediction_index(root, image_stems=None):
    root = Path(root)
    image_stems = list(image_stems or [])
    image_stem_set = set(image_stems)
    index = {}
    duplicate_count = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rank = prediction_candidate_rank(path)
        if rank is None:
            continue

        canonical = canonical_prediction_stem(path)
        numbered_idx = numbered_prediction_index(path)
        if canonical not in image_stem_set and numbered_idx is not None and numbered_idx < len(image_stems):
            canonical = image_stems[numbered_idx]

        if image_stem_set and canonical not in image_stem_set:
            continue

        current = index.get(canonical)
        if current is None or rank < current[0]:
            if current is not None:
                duplicate_count += 1
            index[canonical] = (rank, path)
        else:
            duplicate_count += 1

    if duplicate_count:
        print(f"[{root.name}] ignored {duplicate_count} lower-priority duplicate prediction files.")
    return {stem: path for stem, (_, path) in index.items()}


def collect_pred_dirs(pred_args, pred_root):
    pred_dirs = [parse_pred_arg(v) for v in pred_args or []]
    if pred_root:
        for child in sorted(Path(pred_root).iterdir()):
            if child.is_dir():
                pred_dirs.append((child.name, child))
    if not pred_dirs:
        raise ValueError("Please provide at least one --pred METHOD=dir or --pred-root dir.")
    return pred_dirs


def find_by_stem(root, stem, suffixes=("", "_mask", "_pred", "_segmentation")):
    root = Path(root)
    candidates = []
    for suffix in suffixes:
        for ext in IMAGE_EXTS:
            candidates.append(root / f"{stem}{suffix}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.stem in {stem, f"{stem}_mask", f"{stem}_pred"}:
            return path
    return None


def read_image(path, image_size):
    image = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy((image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1))[None]
    return image, tensor


def read_mask(path, image_size):
    mask = cv2.imread(os.fspath(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    return torch.from_numpy(mask[None, None]), mask.astype(bool)


def binary_metrics(pred, gt, eps=1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.float64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.float64)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return float(dice), float(iou)


@torch.no_grad()
def mask_feature_vector(extractor, image, mask, device):
    feats = extractor.forward_weighted(image.to(device), mask.to(device))
    mean = feats["dinov3_mean"].flatten(1)
    std = feats["dinov3_std"].flatten(1)
    vec = torch.cat([mean, std], dim=1)
    return torch.nn.functional.normalize(vec.float(), dim=1)[0].cpu().numpy()


def feature_distances(pred_vec, gt_vec):
    pred_vec = pred_vec.astype(np.float64)
    gt_vec = gt_vec.astype(np.float64)
    l2 = float(np.linalg.norm(pred_vec - gt_vec))
    cos = float(1.0 - np.dot(pred_vec, gt_vec) / ((np.linalg.norm(pred_vec) * np.linalg.norm(gt_vec)) + 1e-12))
    return l2, cos


def overlay_mask(image_rgb, mask_bool, color):
    image = image_rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    image[mask_bool] = (0.55 * image[mask_bool].astype(np.float32) + 0.45 * color_arr).astype(np.uint8)
    return image


def save_scatter(rows, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping scatter plot.")
        return
    methods = sorted({r["method"] for r in rows})
    plt.figure(figsize=(7.0, 4.8))
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        plt.scatter(
            [float(r["dice"]) for r in subset],
            [float(r["dino_l2"]) for r in subset],
            s=16,
            alpha=0.72,
            label=method,
        )
    plt.xlabel("Dice")
    plt.ylabel("DINOv3 mask-feature distance to GT")
    plt.title("Overlap accuracy versus DINOv3 feature-space mask distance")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_pair_figure(pair, image_rgb, gt_mask, pred_a, pred_b, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping pair figure.")
        return
    panels = [
        ("Image", image_rgb),
        ("GT", gt_mask.astype(np.uint8) * 255),
        (
            f"{pair['method_a']}\nDice={pair['dice_a']:.3f}, DINO={pair['dino_l2_a']:.3f}",
            pred_a.astype(np.uint8) * 255,
        ),
        (
            f"{pair['method_b']}\nDice={pair['dice_b']:.3f}, DINO={pair['dino_l2_b']:.3f}",
            pred_b.astype(np.uint8) * 255,
        ),
        ("GT overlay", overlay_mask(image_rgb, gt_mask, (80, 255, 80))),
        ("Pred A overlay", overlay_mask(image_rgb, pred_a, (255, 80, 80))),
        ("Pred B overlay", overlay_mask(image_rgb, pred_b, (80, 120, 255))),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 2.5))
    for ax, (title, panel) in zip(axes, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"{pair['name']}: similar Dice gap={pair['dice_gap']:.4f}, "
        f"DINO distance ratio={pair['dino_ratio']:.2f}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def find_comparable_pairs(rows, dice_tol, min_pair_dice=0.0):
    by_name = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    pairs = []
    for name, group in by_name.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["method"] == b["method"]:
                    continue
                if min(float(a["dice"]), float(b["dice"])) < min_pair_dice:
                    continue
                dice_gap = abs(float(a["dice"]) - float(b["dice"]))
                if dice_gap > dice_tol:
                    continue
                da, db = float(a["dino_l2"]), float(b["dino_l2"])
                small, large = sorted([max(da, 1e-8), max(db, 1e-8)])
                ratio = large / small
                dino_gap = abs(da - db)
                if da <= db:
                    left, right = a, b
                else:
                    left, right = b, a
                pairs.append(
                    {
                        "name": name,
                        "method_a": left["method"],
                        "method_b": right["method"],
                        "dice_a": float(left["dice"]),
                        "dice_b": float(right["dice"]),
                        "dino_l2_a": float(left["dino_l2"]),
                        "dino_l2_b": float(right["dino_l2"]),
                        "dice_gap": dice_gap,
                        "dino_gap": dino_gap,
                        "dino_ratio": ratio,
                    }
                )
    pairs.sort(key=lambda p: (p["dino_gap"], p["dino_ratio"], -p["dice_gap"]), reverse=True)
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze whether similar Dice masks differ in DINOv3 mask-feature space."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--pred", action="append", default=[], help="METHOD=prediction_dir. Can be repeated.")
    parser.add_argument("--pred-root", default="", help="Directory whose immediate subdirectories are methods.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dinov3-name", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--dinov3-image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dice-tol", type=float, default=0.02)
    parser.add_argument(
        "--min-pair-dice",
        type=float,
        default=0.70,
        help="Only report comparable pairs where both masks have at least this Dice score.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    pred_dirs = collect_pred_dirs(args.pred, args.pred_root)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    image_stems = [p.stem for p in image_paths]
    pred_indexes = {}
    for method, pred_dir in pred_dirs:
        pred_index = build_prediction_index(pred_dir, image_stems=image_stems)
        pred_indexes[method] = pred_index
        print(f"[{method}] indexed {len(pred_index)} prediction masks from {pred_dir}.")

    extractor = DINOv3FeatureExtractor(
        model_name=args.dinov3_name,
        pretrained=True,
        freeze=True,
        image_size=args.dinov3_image_size,
    ).to(device)
    extractor.eval()

    rows = []
    masks_cache = {}
    images_cache = {}
    gt_vec_cache = {}
    for image_path in image_paths:
        stem = image_path.stem
        gt_path = find_by_stem(mask_dir, stem + args.mask_suffix, suffixes=("",))
        if gt_path is None:
            print(f"Skip {stem}: GT mask not found.")
            continue
        image_rgb, image_tensor = read_image(image_path, args.image_size)
        gt_tensor, gt_bool = read_mask(gt_path, args.image_size)
        gt_vec = mask_feature_vector(extractor, image_tensor, gt_tensor, device)
        images_cache[stem] = image_rgb
        masks_cache[(stem, "GT")] = gt_bool
        gt_vec_cache[stem] = gt_vec

        for method, pred_dir in pred_dirs:
            pred_path = pred_indexes[method].get(stem)
            if pred_path is None:
                print(f"Skip {stem}/{method}: prediction not found in {pred_dir}.")
                continue
            pred_tensor, pred_bool = read_mask(pred_path, args.image_size)
            pred_vec = mask_feature_vector(extractor, image_tensor, pred_tensor, device)
            dino_l2, dino_cos = feature_distances(pred_vec, gt_vec)
            dice, iou = binary_metrics(pred_bool, gt_bool)
            masks_cache[(stem, method)] = pred_bool
            rows.append(
                {
                    "name": stem,
                    "method": method,
                    "dice": dice,
                    "iou": iou,
                    "dino_l2": dino_l2,
                    "dino_cos": dino_cos,
                    "pred_area": float(pred_bool.mean()),
                    "gt_area": float(gt_bool.mean()),
                    "pred_path": os.fspath(pred_path),
                }
            )

    csv_path = output_dir / "dinov3_mask_distance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["name", "method", "dice", "iou", "dino_l2", "dino_cos", "pred_area", "gt_area", "pred_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        save_scatter(rows, output_dir / "dice_vs_dinov3_distance.png")

    pairs = find_comparable_pairs(rows, args.dice_tol, args.min_pair_dice)
    pair_csv = output_dir / "comparable_dice_pairs.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "method_a",
            "method_b",
            "dice_a",
            "dice_b",
            "dino_l2_a",
            "dino_l2_b",
            "dice_gap",
            "dino_gap",
            "dino_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    if pairs:
        top = pairs[0]
        save_pair_figure(
            top,
            images_cache[top["name"]],
            masks_cache[(top["name"], "GT")],
            masks_cache[(top["name"], top["method_a"])],
            masks_cache[(top["name"], top["method_b"])],
            output_dir / "top_comparable_pair.png",
        )

    print(f"Wrote {len(rows)} mask measurements to {csv_path}")
    print(f"Wrote {len(pairs)} comparable-Dice pairs to {pair_csv}")
    if pairs:
        print("Top pair:", pairs[0])


if __name__ == "__main__":
    main()
