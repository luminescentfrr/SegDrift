import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image


STEP_RE = re.compile(r"step_(\d+)$")
TRAIN_RE = re.compile(
    r"step=(?P<step>\d+).*?loss=(?P<loss>[-+0-9.eE]+).*?"
    r"drift=(?P<drift>[-+0-9.eE]+).*?seg=(?P<seg>[-+0-9.eE]+).*?"
    r"dice=(?P<dice_loss>[-+0-9.eE]+).*?cls_prior=(?P<cls_prior>[-+0-9.eE]+)"
)
VAL_RE = re.compile(r"validation\s+(?P<body>.*)")


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.facecolor": "#FBFBFB",
            "figure.facecolor": "white",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
        }
    )


def parse_step_dir(path):
    match = STEP_RE.match(path.name)
    return int(match.group(1)) if match else None


def list_sample_steps(samples_dir):
    items = []
    for step_dir in Path(samples_dir).glob("step_*"):
        if not step_dir.is_dir():
            continue
        step = parse_step_dir(step_dir)
        if step is None:
            continue
        overlay = step_dir / "overlay.png"
        prior = step_dir / "prior_heatmap.png"
        if overlay.exists() or prior.exists():
            items.append((step, step_dir, overlay if overlay.exists() else None, prior if prior.exists() else None))
    return sorted(items, key=lambda x: x[0])


def nearest_available_steps(available, requested):
    available_arr = np.asarray(available, dtype=int)
    chosen = []
    for req in requested:
        idx = int(np.argmin(np.abs(available_arr - req)))
        step = int(available_arr[idx])
        if step not in chosen:
            chosen.append(step)
    return chosen


def default_steps(available, max_tiles=6):
    preferred = [500, 1000, 2000, 5000, 10000, 100000]
    max_step = max(available)
    preferred = [s for s in preferred if s <= max_step]
    chosen = nearest_available_steps(available, preferred)
    if len(chosen) >= min(max_tiles, len(available)):
        return chosen[:max_tiles]
    quantile_idx = np.linspace(0, len(available) - 1, min(max_tiles, len(available))).round().astype(int)
    for idx in quantile_idx:
        step = available[int(idx)]
        if step not in chosen:
            chosen.append(step)
    return sorted(chosen[:max_tiles])


def parse_log(log_path):
    train_rows = []
    val_rows = []
    last_step = None
    if log_path is None or not Path(log_path).exists():
        return train_rows, val_rows

    for line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            row = {"step": int(m.group("step"))}
            for key in ("loss", "drift", "seg", "dice_loss", "cls_prior"):
                row[key] = float(m.group(key))
            train_rows.append(row)
            last_step = row["step"]
            continue
        m = VAL_RE.search(line)
        if m and last_step is not None:
            row = {"step": last_step}
            for token in m.group("body").split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
            val_rows.append(row)
    return train_rows, val_rows


def read_image(path):
    if path is None or not Path(path).exists():
        return None
    return Image.open(path).convert("RGB")


def format_step(step):
    if step >= 1000:
        value = step / 1000
        return f"{int(value)}k" if value.is_integer() else f"{value:g}k"
    return str(step)


def detect_prediction_box(img):
    if img is None:
        return None
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    redness = r - 0.55 * g - 0.45 * b
    mask = (r > 120) & (r - g > 28) & (r - b > 28) & (redness > np.percentile(redness, 87))
    frac = float(mask.mean())
    if frac < 0.001 or frac > 0.45:
        return None
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_square_box(box, width, height, scale=2.35):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0) * scale
    side = min(max(side, 32), max(width, height))
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    right = int(round(cx + side / 2))
    bottom = int(round(cy + side / 2))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def infer_lesion_crop_box(sample_items, selected_steps, scale):
    item_by_step = {step: item for step, *item in sample_items}
    for step in reversed(selected_steps):
        if step not in item_by_step:
            continue
        _, overlay_path, _ = item_by_step[step]
        overlay = read_image(overlay_path)
        box = detect_prediction_box(overlay)
        if box is not None:
            width, height = overlay.size
            return expand_square_box(box, width, height, scale=scale)
    return None


def square_tile(img, tile_size, fill="#F7F7F7", crop_frac=1.0, crop_box=None):
    if img is None:
        return Image.new("RGB", (tile_size, tile_size), fill)
    img = img.copy()
    w, h = img.size
    if crop_box is not None:
        left, top, right, bottom = crop_box
        left = max(0, min(w - 1, left))
        top = max(0, min(h - 1, top))
        right = max(left + 1, min(w, right))
        bottom = max(top + 1, min(h, bottom))
        img = img.crop((left, top, right, bottom))
        w, h = img.size
    elif 0 < crop_frac < 1:
        crop_w = max(1, int(w * crop_frac))
        crop_h = max(1, int(h * crop_frac))
        left = max(0, (w - crop_w) // 2)
        top = max(0, (h - crop_h) // 2)
        img = img.crop((left, top, left + crop_w, top + crop_h))
        w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((tile_size, tile_size), Image.BICUBIC)


def smooth_series(y, window):
    if window <= 1 or y.size < 3:
        return y
    window = min(int(window), y.size if y.size % 2 == 1 else y.size - 1)
    if window < 3:
        return y
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def metric_series(train_rows, val_rows, metric):
    if metric.startswith("val_"):
        key = metric[4:]
        rows = [r for r in val_rows if key in r]
    else:
        key = metric
        rows = [r for r in train_rows if key in r]
    if not rows:
        return np.array([]), np.array([])
    return np.asarray([r["step"] for r in rows]), np.asarray([r[key] for r in rows])


def nice_metric_label(metric):
    labels = {
        "loss": "Training loss",
        "drift": "Drifting loss",
        "seg": "Segmentation loss",
        "dice_loss": "Dice loss",
        "cls_prior": "Prior loss",
        "val_dice": "Val. Dice",
        "val_iou": "Val. IoU",
        "val_miou": "Val. mIoU",
        "val_recall": "Val. Recall",
        "val_hd95": "Validation HD95",
    }
    return labels.get(metric, metric)


def build_figure(
    sample_items,
    selected_steps,
    train_rows,
    val_rows,
    out_prefix,
    metric,
    title,
    tile_size,
    xscale,
    ymin=None,
    ymax=None,
    crop_frac=0.84,
    crop_mode="lesion",
    lesion_crop_scale=2.35,
    phase_bands=True,
    fig_width=7.16,
    smooth_window=7,
):
    item_by_step = {step: item for step, *item in sample_items}
    selected_steps = [s for s in selected_steps if s in item_by_step]
    if not selected_steps:
        raise RuntimeError("No selected steps have saved overlay/prior images.")

    n = len(selected_steps)
    fig_w = fig_width
    fig_h = 3.02 if n <= 8 else 3.35
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(3, n, height_ratios=[1.0, 1.0, 0.78], hspace=0.06, wspace=0.035)
    lesion_crop_box = None
    if crop_mode == "lesion":
        lesion_crop_box = infer_lesion_crop_box(sample_items, selected_steps, scale=lesion_crop_scale)

    for col, step in enumerate(selected_steps):
        step_dir, overlay_path, prior_path = item_by_step[step]
        prior = read_image(prior_path)
        overlay = read_image(overlay_path)

        for row, img in enumerate((prior, overlay)):
            ax_tile = fig.add_subplot(gs[row, col])
            tile = square_tile(
                img,
                tile_size=tile_size,
                crop_frac=1.0 if crop_mode == "none" else crop_frac,
                crop_box=lesion_crop_box,
            )
            ax_tile.imshow(tile)
            ax_tile.set_xticks([])
            ax_tile.set_yticks([])
            if row == 0:
                ax_tile.set_title(format_step(step), pad=2.0, fontsize=6.9, color="#222222")
            for spine in ax_tile.spines.values():
                spine.set_visible(True)
                spine.set_color("#FFFFFF")
                spine.set_linewidth(1.0)

    ax = fig.add_subplot(gs[2, :])
    x, y = metric_series(train_rows, val_rows, metric)
    if x.size:
        color = "#2B6CB0" if metric.startswith("val_") else "#2F2F2F"
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        y_smooth = smooth_series(y, smooth_window)
        low = float(np.nanmin(y)) if ymin is None else ymin
        high = float(np.nanmax(y)) if ymax is None else ymax
        pad = max((high - low) * 0.16, 0.02)
        if ymin is None:
            low = max(0.0, low - pad)
        if ymax is None and metric.startswith("val_"):
            high = min(1.0, high + pad * 0.55)
        elif ymax is None:
            high = high + pad

        if xscale == "log":
            ax.set_xscale("log")
        if not metric.startswith("val_") and y.min() > 0:
            ax.set_yscale("log")
        xmax_for_bands = max(max(selected_steps), x.max())
        if phase_bands and xscale == "log":
            bands = [
                (min(selected_steps), min(1000, xmax_for_bands), "#FFF0DD"),
                (1000, min(5000, xmax_for_bands), "#EAF2FF"),
                (5000, xmax_for_bands, "#F3F4F6"),
            ]
            for left, right, face in bands:
                if right > left:
                    ax.axvspan(left, right, color=face, alpha=0.55, linewidth=0, zorder=-3)
        if smooth_window > 1:
            ax.plot(x, y, color=color, linewidth=0.65, alpha=0.22, zorder=1.5)
        ax.fill_between(x, low, y_smooth, color=color, alpha=0.095, linewidth=0, zorder=1)
        ax.plot(x, y_smooth, color=color, linewidth=1.65, zorder=2)
        y_interp = np.interp(selected_steps, x, y_smooth)
        ax.scatter(selected_steps, y_interp, s=20, color="#E97823", edgecolor="white", linewidth=0.5, zorder=3)
        for step in selected_steps:
            ax.axvline(step, color="#CFD4DC", linewidth=0.35, alpha=0.55, zorder=0)
        ax.set_ylim(low, high)
    else:
        ax.text(0.5, 0.5, f"No `{metric}` series found in log.", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Training step", labelpad=2)
    ax.set_ylabel(nice_metric_label(metric), labelpad=4)
    ax.grid(True, axis="y", color="#D9DEE7", linewidth=0.45, alpha=0.8)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", labelsize=6.6, pad=1.5)
    if selected_steps:
        xmax = max(max(selected_steps), x.max() if x.size else selected_steps[-1])
        if xscale == "log":
            xmin = max(1, min(selected_steps) * 0.75)
            ax.set_xlim(xmin, xmax * 1.12)
            ax.set_xticks(selected_steps)
            ax.set_xticklabels([format_step(s) for s in selected_steps], fontsize=6.2)
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        else:
            ax.set_xlim(0, xmax * 1.02)

    if title:
        fig.suptitle(title, y=0.985, fontsize=8.3)
    fig.text(
        0.079,
        0.958,
        "a",
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.079,
        0.335,
        "b",
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.036,
        0.715,
        "Prior",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
        color="#333333",
    )
    fig.text(
        0.036,
        0.500,
        "Prediction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
        color="#333333",
    )
    fig.subplots_adjust(left=0.082, right=0.998, top=0.895 if title else 0.945, bottom=0.145)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build a SegDrift training-dynamics figure from saved validation samples.")
    parser.add_argument("--workdir", required=True, help="Training run directory containing log.txt and samples/step_*/.")
    parser.add_argument("--samples-dir", default=None, help="Override samples directory. Defaults to WORKDIR/samples.")
    parser.add_argument("--log", default=None, help="Override log path. Defaults to WORKDIR/log.txt.")
    parser.add_argument("--out", required=True, help="Output prefix, e.g. paper/figures/training_dynamics.")
    parser.add_argument("--steps", default="", help="Comma-separated steps to show. Nearest available saved step is used.")
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--metric", default="val_dice", help="Curve metric: val_dice, val_iou, loss, drift, seg, dice_loss, cls_prior.")
    parser.add_argument("--title", default="")
    parser.add_argument("--tile-size", type=int, default=180)
    parser.add_argument("--xscale", choices=["log", "linear"], default="log")
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--crop-frac", type=float, default=0.84, help="Center crop fraction for snapshot tiles; use 1.0 to disable.")
    parser.add_argument("--crop-mode", choices=["lesion", "center", "none"], default="lesion")
    parser.add_argument("--lesion-crop-scale", type=float, default=2.35)
    parser.add_argument("--fig-width", type=float, default=7.16, help="Figure width in inches, suitable for a two-column IEEE figure.")
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--no-phase-bands", action="store_true", help="Disable subtle background bands on the dynamics curve.")
    args = parser.parse_args()

    setup_style()
    workdir = Path(args.workdir)
    samples_dir = Path(args.samples_dir) if args.samples_dir else workdir / "samples"
    log_path = Path(args.log) if args.log else workdir / "log.txt"

    sample_items = list_sample_steps(samples_dir)
    if not sample_items:
        raise RuntimeError(f"No sample directories found under {samples_dir}")
    available = [item[0] for item in sample_items]
    if args.steps.strip():
        requested = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
        selected_steps = nearest_available_steps(available, requested)
    else:
        selected_steps = default_steps(available, max_tiles=args.max_tiles)

    train_rows, val_rows = parse_log(log_path)
    print(f"Found {len(sample_items)} saved sample steps: {available[0]} -> {available[-1]}")
    print(f"Selected steps: {selected_steps}")
    print(f"Parsed {len(train_rows)} train log rows and {len(val_rows)} validation rows from {log_path}")
    build_figure(
        sample_items,
        selected_steps,
        train_rows,
        val_rows,
        args.out,
        args.metric,
        args.title,
        args.tile_size,
        args.xscale,
        args.ymin,
        args.ymax,
        args.crop_frac,
        args.crop_mode,
        args.lesion_crop_scale,
        not args.no_phase_bands,
        args.fig_width,
        args.smooth_window,
    )
    print(f"Saved {Path(args.out).with_suffix('.png')}, .pdf, and .svg")


if __name__ == "__main__":
    main()
