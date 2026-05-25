# -*- coding: utf-8 -*-
"""
Interactively select seed points on a grayscale image.

Usage:
  python Select_Seed_Points.py

In the matplotlib window:
  - Left click each target object center.
  - Press Enter when finished.
  - Close the window if you want to cancel.

Outputs:
  outputs/output_seed_points.csv
  outputs/output_seed_points_preview.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

from Segmentation_ import clip_roi, display_image_for_review


INPUT_FILE = Path("image.tif")
OUTPUT_FOLDER = Path("outputs")
SEED_CSV = OUTPUT_FOLDER / "output_seed_points.csv"
SEED_PREVIEW = OUTPUT_FOLDER / "output_seed_points_preview.png"
SEED_ZOOM_PREVIEW = OUTPUT_FOLDER / "output_seed_points_zoom_preview.png"

# Keep this synchronized with the ROI in Segmentation_.py.
ROI = (370, 260, 180, 178)
ZOOM_TO_ROI = True


def save_seed_csv(path: Path, points: list[tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x_px", "y_px"])
        writer.writeheader()
        for index, (x, y) in enumerate(points, start=1):
            writer.writerow({"id": index, "x_px": x, "y_px": y})


def save_preview(path: Path, display: np.ndarray, points: list[tuple[float, float]], roi_box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = roi_box
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    ax.imshow(display, cmap="gray")
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="cyan",
            linewidth=1.0,
        )
    )
    for index, (x, y) in enumerate(points, start=1):
        ax.plot(x, y, marker="+", color="lime", markersize=7, markeredgewidth=1.2)
        ax.text(x + 3, y - 3, str(index), color="lime", fontsize=7)
    ax.set_axis_off()
    ax.set_title("Clicked seed points")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_zoom_preview(path: Path, display: np.ndarray, points: list[tuple[float, float]], roi_box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = roi_box
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    ax.imshow(
        display[y0:y1, x0:x1],
        cmap="gray",
        extent=(x0, x1, y1, y0),
        interpolation="nearest",
    )
    for index, (x, y) in enumerate(points, start=1):
        ax.plot(x, y, marker="+", color="lime", markersize=9, markeredgewidth=1.4)
        ax.text(x + 1, y - 1, str(index), color="lime", fontsize=8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_axis_off()
    ax.set_title("Clicked seed points - zoomed ROI")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image = tiff.imread(INPUT_FILE).astype(np.float32)
    display = display_image_for_review(image)
    roi_box = clip_roi(ROI, image.shape)
    x0, y0, x1, y1 = roi_box

    if ZOOM_TO_ROI:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        ax.imshow(
            display[y0:y1, x0:x1],
            cmap="gray",
            extent=(x0, x1, y1, y0),
            interpolation="nearest",
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title("Zoomed ROI: left click target centers, then press Enter")
    else:
        fig, ax = plt.subplots(figsize=(12, 9), dpi=120)
        ax.imshow(display, cmap="gray")
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="cyan",
                linewidth=1.0,
            )
        )
        ax.set_title("Left click target centers, then press Enter")
    ax.set_axis_off()

    print("Left click each target object center in the image window.")
    print("Use the matplotlib toolbar magnifier/pan tools if you want to zoom more.")
    print("Press Enter in the image window when finished.")
    points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    points = [(float(x), float(y)) for x, y in points]
    save_seed_csv(SEED_CSV, points)
    save_preview(SEED_PREVIEW, display, points, roi_box)
    save_zoom_preview(SEED_ZOOM_PREVIEW, display, points, roi_box)

    print(f"Saved {len(points)} seed points: {SEED_CSV}")
    print(f"Preview: {SEED_PREVIEW}")
    print(f"Zoom preview: {SEED_ZOOM_PREVIEW}")


if __name__ == "__main__":
    main()
