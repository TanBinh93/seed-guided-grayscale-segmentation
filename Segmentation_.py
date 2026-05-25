# -*- coding: utf-8 -*-
"""
Seed-guided segmentation of small bright objects in a grayscale image.

Default outputs:
  outputs/output_coordinates.csv
  outputs/output_mask.tif
  outputs/output_labels.tif
  outputs/output_overlay.png
  outputs/output_object_intensity_summary.csv

Coordinates: x = column, y = row, origin at the top-left corner of the image.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, morphology, segmentation


def finite_values(image: np.ndarray) -> np.ndarray:
    """Return finite pixels as a 1D array."""
    return image[np.isfinite(image)]


def fill_nonfinite_with_median(image: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf pixels before filters that would otherwise propagate them."""
    image = image.astype(np.float32, copy=True)
    finite = finite_values(image)
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    image[~np.isfinite(image)] = float(np.median(finite))
    return image


def robust_percentile_range(
    image: np.ndarray,
    percentiles: tuple[float, float],
    fallback: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Compute percentile limits with a safe fallback for empty/constant data."""
    finite = finite_values(image)
    if finite.size == 0:
        return fallback
    lo, hi = np.percentile(finite, percentiles)
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return fallback
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def robust_rescale(
    image: np.ndarray,
    percentiles: tuple[float, float] | None = None,
    in_range: tuple[float, float] | str | None = None,
) -> np.ndarray:
    """Rescale to [0, 1] while avoiding NaN/Inf propagation."""
    clean = fill_nonfinite_with_median(image)
    if percentiles is not None:
        scale_range = robust_percentile_range(clean, percentiles)
    elif in_range == "image":
        scale_range = robust_percentile_range(clean, (0.0, 100.0))
    elif in_range is not None:
        scale_range = in_range
    else:
        scale_range = robust_percentile_range(clean, (1.0, 99.8))
    out = exposure.rescale_intensity(clean, in_range=scale_range, out_range=(0.0, 1.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Method summary:
# 1. Process only a region of interest around the target bright objects.
# 2. Reduce fine noise with a light Gaussian smoothing filter.
# 3. Estimate the local background with a larger Gaussian filter and subtract it
#    to enhance small bright objects.
# 4. Add a morphological white top-hat response, which is useful for detecting
#    small bright structures on a slowly varying background. Several disk sizes
#    are used so that both dot-like and small cluster-like objects can respond.
# 5. Add a ridge/filament response to recover weak elongated or irregular
#    target clusters that are not well modeled as round blobs.
# 6. Add a local z-score response to recover weak objects that are still
#    brighter than their local neighborhood.
# 7. Combine these responses into a robust score image, then apply automatic and
#    percentile-based thresholding.
# 8. Add LoG (Laplacian of Gaussian) blob detection to recover low-contrast small
#    blobs.
# 9. Clean the binary mask with morphological operations and separate touching
#    objects with watershed.
# 10. Filter objects by area and eccentricity, then export their centroids.


def clip_roi(roi: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = roi
    image_height, image_width = shape
    x0 = max(0, min(x, image_width - 1))
    y0 = max(0, min(y, image_height - 1))
    x1 = max(x0 + 1, min(x0 + width, image_width))
    y1 = max(y0 + 1, min(y0 + height, image_height))
    return x0, y0, x1, y1


def enhanced_small_bright_objects(
    roi_image: np.ndarray, smooth_sigma: float, background_sigma: float
) -> np.ndarray:
    """Enhance small bright objects by subtracting the local background."""
    roi_image = fill_nonfinite_with_median(roi_image)
    smooth = filters.gaussian(roi_image, sigma=smooth_sigma, preserve_range=True)
    background = filters.gaussian(smooth, sigma=background_sigma, preserve_range=True)
    enhanced = smooth - background
    return robust_rescale(enhanced, in_range="image")


def robust_small_object_score(
    roi_image: np.ndarray,
    smooth_sigma: float,
    background_sigma: float,
    tophat_radii: tuple[int, ...],
    ridge_sigmas: tuple[float, ...],
    ridge_weight: float,
    score_mode: str,
) -> np.ndarray:
    """Combine several enhancement maps for weak bright objects."""
    roi_image = fill_nonfinite_with_median(roi_image)
    smooth = filters.gaussian(roi_image, sigma=smooth_sigma, preserve_range=True)
    background = filters.gaussian(smooth, sigma=background_sigma, preserve_range=True)
    residual = smooth - background
    if score_mode == "phase_absolute":
        residual_for_score = np.abs(residual)
    else:
        residual_for_score = residual
    residual_score = robust_rescale(residual_for_score, percentiles=(1.0, 99.8))

    normalized = robust_rescale(roi_image, percentiles=(1.0, 99.8))
    tophat_maps = [
        morphology.white_tophat(normalized, morphology.disk(radius))
        for radius in tophat_radii
    ]
    if score_mode == "phase_absolute":
        black_tophat_maps = [
            morphology.black_tophat(normalized, morphology.disk(radius))
            for radius in tophat_radii
        ]
        tophat = np.maximum(np.maximum.reduce(tophat_maps), np.maximum.reduce(black_tophat_maps))
    else:
        tophat = np.maximum.reduce(tophat_maps)
    tophat_score = robust_rescale(tophat, percentiles=(1.0, 99.8))

    ridge_bright = filters.sato(
        normalized,
        sigmas=ridge_sigmas,
        black_ridges=False,
        mode="reflect",
    )
    if score_mode == "phase_absolute":
        ridge_dark = filters.sato(
            normalized,
            sigmas=ridge_sigmas,
            black_ridges=True,
            mode="reflect",
        )
        ridge = np.maximum(ridge_bright, ridge_dark)
    else:
        ridge = ridge_bright
    ridge_score = robust_rescale(ridge, percentiles=(1.0, 99.8))

    local_mean = filters.gaussian(roi_image, sigma=background_sigma, preserve_range=True)
    local_sq_mean = filters.gaussian(roi_image * roi_image, sigma=background_sigma, preserve_range=True)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 1e-8))
    if score_mode == "phase_absolute":
        z_score = np.abs((roi_image - local_mean) / local_std)
    else:
        z_score = np.clip((roi_image - local_mean) / local_std, 0.0, None)
    z_score = robust_rescale(z_score, percentiles=(1.0, 99.8))

    score = np.maximum.reduce(
        [
            residual_score,
            tophat_score,
            z_score,
            ridge_weight * ridge_score,
        ]
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def automatic_threshold(image: np.ndarray, method: str) -> float:
    if method == "otsu":
        return float(filters.threshold_otsu(image))
    if method == "yen":
        return float(filters.threshold_yen(image))
    if method == "li":
        return float(filters.threshold_li(image))
    raise ValueError(f"Unknown thresholding method: {method}")


def split_touching_objects(mask: np.ndarray, score: np.ndarray | None = None) -> np.ndarray:
    """Separe legerement les taches qui se touchent avec watershed."""
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(mask)
    marker_image = distance if score is None else filters.gaussian(score, sigma=0.8)
    local_max = morphology.local_maxima(marker_image) & mask
    markers = measure.label(local_max)
    if markers.max() == 0:
        markers = measure.label(mask)
    return segmentation.watershed(-distance, markers=markers, mask=mask)


def read_seed_points(path: str | None) -> list[tuple[float, float]]:
    """Read user-clicked seed points as full-image (x, y) coordinates."""
    if path is None:
        return []

    seed_path = Path(path)
    if not seed_path.exists():
        return []

    seeds: list[tuple[float, float]] = []
    with seed_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seeds.append((float(row["x_px"]), float(row["y_px"])))
    return seeds


def seed_guidance_masks(
    score: np.ndarray,
    enhanced: np.ndarray,
    seeds: list[tuple[float, float]],
    roi_origin: tuple[int, int],
    seed_radius: int,
    seed_local_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build seed-anchored candidate components around clicked points."""
    x0, y0 = roi_origin
    yy, xx = np.ogrid[: score.shape[0], : score.shape[1]]
    candidate = np.zeros(score.shape, dtype=bool)
    restriction = np.zeros(score.shape, dtype=bool)

    for x_seed, y_seed in seeds:
        x = int(round(x_seed - x0))
        y = int(round(y_seed - y0))
        if x < 0 or x >= score.shape[1] or y < 0 or y >= score.shape[0]:
            continue

        disk = (xx - x) ** 2 + (yy - y) ** 2 <= seed_radius * seed_radius
        restriction |= disk
        seed_core = (xx - x) ** 2 + (yy - y) ** 2 <= 1 * 1

        y_min = max(0, y - seed_radius)
        y_max = min(score.shape[0], y + seed_radius + 1)
        x_min = max(0, x - seed_radius)
        x_max = min(score.shape[1], x + seed_radius + 1)

        response = np.maximum(score, enhanced)
        local_response = response[y_min:y_max, x_min:x_max]
        local_disk = disk[y_min:y_max, x_min:x_max]
        if np.any(local_disk):
            local_response = local_response[local_disk]
        response_threshold = np.percentile(local_response, seed_local_percentile)
        local_candidate = disk & ((response >= response_threshold) | seed_core)
        local_labels = measure.label(local_candidate)
        seed_label = local_labels[y, x]
        if seed_label == 0:
            candidate |= seed_core
        else:
            candidate |= local_labels == seed_label

    if np.any(candidate):
        growth_radius = seed_radius + 1
        growth_percentile = max(50.0, seed_local_percentile - 13.0)
        response = np.maximum(score, enhanced)
        growth_zone = np.zeros(score.shape, dtype=bool)
        for x_seed, y_seed in seeds:
            x = int(round(x_seed - x0))
            y = int(round(y_seed - y0))
            if x < 0 or x >= score.shape[1] or y < 0 or y >= score.shape[0]:
                continue
            growth_zone |= (xx - x) ** 2 + (yy - y) ** 2 <= growth_radius * growth_radius

        if np.any(growth_zone):
            response_threshold = np.percentile(response[growth_zone], growth_percentile)
            allowed = growth_zone & (response >= response_threshold)
            grown = candidate.copy()
            for _ in range(growth_radius):
                next_grown = grown | (morphology.binary_dilation(grown, morphology.disk(1)) & allowed)
                if np.array_equal(next_grown, grown):
                    break
                grown = next_grown
            candidate = morphology.remove_small_holes(grown, area_threshold=8)

    return candidate, restriction


def split_seed_guided_objects(
    mask: np.ndarray,
    seeds: list[tuple[float, float]],
    roi_origin: tuple[int, int],
) -> np.ndarray:
    """Split a seed-guided mask into seed-anchored objects."""
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)

    x0, y0 = roi_origin
    markers = np.zeros(mask.shape, dtype=np.int32)
    marker_id = 0
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    for x_seed, y_seed in seeds:
        x = int(round(x_seed - x0))
        y = int(round(y_seed - y0))
        if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
            continue
        marker_id += 1
        seed_core = (xx - x) ** 2 + (yy - y) ** 2 <= 1
        markers[seed_core] = marker_id
        mask[seed_core] = True

    if marker_id == 0:
        return measure.label(mask)

    distance = ndi.distance_transform_edt(mask)
    return segmentation.watershed(-distance, markers=markers, mask=mask)


def segment_objects(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold_method: str,
    threshold_scale: float,
    min_area: int,
    max_area: int,
    max_eccentricity: float,
    smooth_sigma: float,
    background_sigma: float,
    score_percentile: float,
    log_threshold: float,
    tophat_radii: tuple[int, ...],
    ridge_sigmas: tuple[float, ...],
    ridge_weight: float,
    ignore_bottom_pixels: int,
    score_mode: str,
    seed_points: list[tuple[float, float]] | None = None,
    seed_radius: int = 10,
    seed_local_percentile: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    x0, y0, x1, y1 = clip_roi(roi, image.shape)
    roi_image = image[y0:y1, x0:x1].astype(np.float32, copy=False)
    enhanced = enhanced_small_bright_objects(roi_image, smooth_sigma, background_sigma)
    score = robust_small_object_score(
        roi_image=roi_image,
        smooth_sigma=smooth_sigma,
        background_sigma=background_sigma,
        tophat_radii=tophat_radii,
        ridge_sigmas=ridge_sigmas,
        ridge_weight=ridge_weight,
        score_mode=score_mode,
    )

    automatic = automatic_threshold(enhanced, threshold_method) * threshold_scale
    sensitive = np.percentile(score, score_percentile)
    raw_mask = (enhanced > automatic) | (score > sensitive)

    blobs = feature.blob_log(
        score,
        min_sigma=1.0,
        max_sigma=5.0,
        num_sigma=8,
        threshold=log_threshold,
        overlap=0.4,
    )
    blob_seed_mask = np.zeros(raw_mask.shape, dtype=bool)
    for y_blob, x_blob, sigma in blobs:
        radius = max(2, int(round(float(sigma) * np.sqrt(2.0))))
        yy, xx = np.ogrid[: raw_mask.shape[0], : raw_mask.shape[1]]
        blob_seed_mask |= (yy - y_blob) ** 2 + (xx - x_blob) ** 2 <= radius * radius

    raw_mask |= blob_seed_mask & (score > np.percentile(score, max(score_percentile - 8.0, 80.0)))
    if seed_points:
        seed_candidate, seed_restriction = seed_guidance_masks(
            score=score,
            enhanced=enhanced,
            seeds=seed_points,
            roi_origin=(x0, y0),
            seed_radius=seed_radius,
            seed_local_percentile=seed_local_percentile,
        )
        if np.any(seed_restriction):
            raw_mask = seed_candidate

    if not seed_points:
        raw_mask = morphology.binary_closing(raw_mask, morphology.disk(1))
    raw_mask = morphology.remove_small_objects(raw_mask, min_size=min_area)
    raw_mask = morphology.remove_small_holes(raw_mask, area_threshold=min_area)
    if ignore_bottom_pixels > 0:
        raw_mask[-ignore_bottom_pixels:, :] = False

    if seed_points:
        roi_labels = split_seed_guided_objects(raw_mask, seed_points, roi_origin=(x0, y0))
    else:
        roi_labels = split_touching_objects(raw_mask, score=score)
    full_labels = np.zeros(image.shape, dtype=np.uint16)
    full_mask = np.zeros(image.shape, dtype=np.uint8)

    rows: list[dict[str, float]] = []
    kept_label = 0
    for region in measure.regionprops(roi_labels, intensity_image=roi_image):
        if not (min_area <= region.area <= max_area):
            continue
        if not seed_points and region.eccentricity > max_eccentricity:
            continue
        region_score = score[roi_labels == region.label]
        if not seed_points and float(np.percentile(region_score, 90)) < 0.18:
            continue

        kept_label += 1
        region_mask = roi_labels == region.label
        full_labels[y0:y1, x0:x1][region_mask] = kept_label
        full_mask[y0:y1, x0:x1][region_mask] = 255

        cy, cx = region.centroid
        min_row, min_col, max_row, max_col = region.bbox
        rows.append(
            {
                "id": kept_label,
                "x_centroid_px": x0 + cx,
                "y_centroid_px": y0 + cy,
                "area_px": float(region.area),
                "equivalent_diameter_px": float(region.equivalent_diameter_area),
                "mean_absorption": float(region.mean_intensity),
                "max_absorption": float(region.max_intensity),
                "eccentricity": float(region.eccentricity),
                "bbox_x0": x0 + min_col,
                "bbox_y0": y0 + min_row,
                "bbox_x1": x0 + max_col,
                "bbox_y1": y0 + max_row,
            }
        )

    return full_mask, full_labels, score, rows


def save_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "id",
        "x_centroid_px",
        "y_centroid_px",
        "area_px",
        "equivalent_diameter_px",
        "mean_absorption",
        "max_absorption",
        "eccentricity",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def finite_region_values(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    values = image[region & np.isfinite(image)]
    if values.size == 0:
        return np.array([0.0], dtype=np.float32)
    return values


def save_object_intensity_summary(
    path: Path,
    labels: np.ndarray,
    ab: np.ndarray,
    df: np.ndarray,
    ph: np.ndarray,
) -> None:
    fieldnames = [
        "id",
        "area_px",
        "x_centroid_px",
        "y_centroid_px",
        "mean_absorption",
        "mean_darkfield",
        "mean_phase",
        "max_absorption",
        "max_darkfield",
        "max_phase",
    ]

    rows = []
    for label_id in sorted(int(v) for v in np.unique(labels) if v != 0):
        region = labels == label_id
        yy, xx = np.nonzero(region)
        ab_values = finite_region_values(ab, region)
        df_values = finite_region_values(df, region)
        ph_values = finite_region_values(ph, region)
        rows.append(
            {
                "id": label_id,
                "area_px": int(region.sum()),
                "x_centroid_px": float(np.mean(xx)),
                "y_centroid_px": float(np.mean(yy)),
                "mean_absorption": float(np.mean(ab_values)),
                "mean_darkfield": float(np.mean(df_values)),
                "mean_phase": float(np.mean(ph_values)),
                "max_absorption": float(np.max(ab_values)),
                "max_darkfield": float(np.max(df_values)),
                "max_phase": float(np.max(ph_values)),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pixel_intensities(
    path: Path,
    labels: np.ndarray,
    ab: np.ndarray,
    df: np.ndarray,
    ph: np.ndarray,
) -> None:
    fieldnames = [
        "label_id",
        "x_px",
        "y_px",
        "absorption",
        "darkfield",
        "phase",
    ]
    yy, xx = np.nonzero(labels > 0)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for y, x in zip(yy, xx):
            ab_value = ab[y, x] if np.isfinite(ab[y, x]) else 0.0
            df_value = df[y, x] if np.isfinite(df[y, x]) else 0.0
            ph_value = ph[y, x] if np.isfinite(ph[y, x]) else 0.0
            writer.writerow(
                {
                    "label_id": int(labels[y, x]),
                    "x_px": int(x),
                    "y_px": int(y),
                    "absorption": float(ab_value),
                    "darkfield": float(df_value),
                    "phase": float(ph_value),
                }
            )


def save_segmentation_intensity_csvs(
    output_dir: Path,
    labels: np.ndarray,
    absorption_file: str,
    darkfield_file: str,
    phase_file: str,
) -> None:
    ab = tiff.imread(absorption_file).astype(np.float32)
    df = tiff.imread(darkfield_file).astype(np.float32)
    ph = tiff.imread(phase_file).astype(np.float32)

    if ab.shape != df.shape or ab.shape != ph.shape or ab.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch for intensity CSVs: ab={ab.shape}, df={df.shape}, "
            f"ph={ph.shape}, labels={labels.shape}"
        )

    save_object_intensity_summary(output_dir / "output_object_intensity_summary.csv", labels, ab, df, ph)
    save_pixel_intensities(output_dir / "output_pixel_intensities.csv", labels, ab, df, ph)

def display_image_for_review(image: np.ndarray) -> np.ndarray:
    """Convert the raw image to a stable grayscale display image."""
    return robust_rescale(image, percentiles=(1.0, 99.7))


def percentile_rescale(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """Robustly rescale an image to [0, 1] using percentile limits."""
    return robust_rescale(image, percentiles=(low_percentile, high_percentile))


def enhance_phase_contrast_for_segmentation(
    image: np.ndarray,
    background_sigma: float,
    clahe_kernel_size: int,
    clahe_clip_limit: float,
) -> np.ndarray:
    """
    Enhance phase-contrast data before segmentation.

    The raw phase image can have a strong low-frequency background that hides
    weak target objects. This preprocessing flattens the background and then
    increases local contrast so small bright structures become detectable.
    """
    image = image.astype(np.float32, copy=False)
    filled = fill_nonfinite_with_median(image)

    denoised = filters.gaussian(filled, sigma=1.0, preserve_range=True)
    background = filters.gaussian(denoised, sigma=background_sigma, preserve_range=True)
    flattened = denoised - background
    flattened = percentile_rescale(flattened, 1.0, 99.8)

    clahe = exposure.equalize_adapthist(
        flattened,
        kernel_size=clahe_kernel_size,
        clip_limit=clahe_clip_limit,
    ).astype(np.float32)

    enhanced = np.maximum(flattened, clahe)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def preprocess_image_for_segmentation(
    image: np.ndarray,
    preprocess_mode: str,
    phase_background_sigma: float,
    phase_clahe_kernel_size: int,
    phase_clahe_clip_limit: float,
) -> np.ndarray:
    if preprocess_mode == "none":
        return image.astype(np.float32, copy=False)
    if preprocess_mode == "phase_contrast_enhanced":
        return enhance_phase_contrast_for_segmentation(
            image=image,
            background_sigma=phase_background_sigma,
            clahe_kernel_size=phase_clahe_kernel_size,
            clahe_clip_limit=phase_clahe_clip_limit,
        )
    raise ValueError(f"Unknown preprocess mode: {preprocess_mode}")


def save_original_review_image(
    path: Path,
    image: np.ndarray,
    roi_box: tuple[int, int, int, int],
    image_name: str,
) -> None:
    x0, y0, x1, y1 = roi_box
    display = display_image_for_review(image)

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
    ax.set_axis_off()
    ax.set_title(f"Original image - {image_name}")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_review_image(
    path: Path,
    image: np.ndarray,
    roi_box: tuple[int, int, int, int],
    title: str,
) -> None:
    x0, y0, x1, y1 = roi_box
    display = display_image_for_review(image)

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
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_overlay(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    rows: list[dict[str, float]],
    roi_box: tuple[int, int, int, int],
    image_name: str,
    draw_numbers: bool,
) -> None:
    x0, y0, x1, y1 = roi_box
    display = display_image_for_review(image)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    ax.imshow(display, cmap="gray")
    ax.contour(mask > 0, levels=[0.5], colors="red", linewidths=0.8)
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

    if draw_numbers:
        for row in rows:
            x = row["x_centroid_px"]
            y = row["y_centroid_px"]
            ax.plot(x, y, marker="+", color="yellow", markersize=5, markeredgewidth=0.9)
            ax.text(x + 3, y - 3, str(int(row["id"])), color="yellow", fontsize=6)

    ax.set_axis_off()
    if draw_numbers:
        ax.set_title(f"Numbered segmentation - {image_name}")
    else:
        ax.set_title(f"Segmentation - {image_name}")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main(
    input_file: str,
    output_folder: str,
    roi: tuple[int, int, int, int],
    threshold_method: str,
    score_percentile: float,
    log_threshold: float,
    threshold_scale: float,
    min_area: int,
    max_area: int,
    max_eccentricity: float,
    smooth_sigma: float,
    background_sigma: float,
    tophat_radii: tuple[int, ...],
    ridge_sigmas: tuple[float, ...],
    ridge_weight: float,
    ignore_bottom_pixels: int,
    preprocess_mode: str,
    score_mode: str,
    seed_points_file: str | None,
    seed_radius: int,
    seed_local_percentile: float,
    absorption_intensity_file: str,
    darkfield_intensity_file: str,
    phase_intensity_file: str,
    phase_background_sigma: float,
    phase_clahe_kernel_size: int,
    phase_clahe_clip_limit: float,
) -> None:
    input_path = Path(input_file)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_image = tiff.imread(input_path).astype(np.float32)
    if raw_image.ndim != 2:
        raise ValueError(f"L'image doit etre 2D, shape trouvee: {raw_image.shape}")

    image = preprocess_image_for_segmentation(
        image=raw_image,
        preprocess_mode=preprocess_mode,
        phase_background_sigma=phase_background_sigma,
        phase_clahe_kernel_size=phase_clahe_kernel_size,
        phase_clahe_clip_limit=phase_clahe_clip_limit,
    )

    roi_box = clip_roi(roi, image.shape)
    seed_points = read_seed_points(seed_points_file)
    mask, labels, _enhanced, rows = segment_objects(
        image=image,
        roi=roi,
        threshold_method=threshold_method,
        threshold_scale=threshold_scale,
        min_area=min_area,
        max_area=max_area,
        max_eccentricity=max_eccentricity,
        smooth_sigma=smooth_sigma,
        background_sigma=background_sigma,
        score_percentile=score_percentile,
        log_threshold=log_threshold,
        tophat_radii=tophat_radii,
        ridge_sigmas=ridge_sigmas,
        ridge_weight=ridge_weight,
        ignore_bottom_pixels=ignore_bottom_pixels,
        score_mode=score_mode,
        seed_points=seed_points,
        seed_radius=seed_radius,
        seed_local_percentile=seed_local_percentile,
    )

    tiff.imwrite(output_dir / "output_mask.tif", mask)
    tiff.imwrite(output_dir / "output_labels.tif", labels)
    tiff.imwrite(output_dir / "output_score.tif", _enhanced.astype(np.float32))
    tiff.imwrite(output_dir / "enhanced_for_segmentation.tif", image.astype(np.float32))
    save_csv(output_dir / "output_coordinates.csv", rows)
    save_segmentation_intensity_csvs(
        output_dir,
        labels,
        absorption_intensity_file,
        darkfield_intensity_file,
        phase_intensity_file,
    )
    save_original_review_image(output_dir / "original.png", raw_image, roi_box, input_path.name)
    save_review_image(
        output_dir / "enhanced.png",
        image,
        roi_box,
        title=f"Enhanced image used for segmentation - {input_path.name}",
    )
    save_overlay(
        output_dir / "output_overlay.png",
        image,
        mask,
        rows,
        roi_box,
        input_path.name,
        draw_numbers=False,
    )
    save_overlay(
        output_dir / "output_overlay_numbered.png",
        image,
        mask,
        rows,
        roi_box,
        input_path.name,
        draw_numbers=True,
    )

    print(f"Image: {input_path}")
    print(f"ROI used: x={roi_box[0]}, y={roi_box[1]}, w={roi_box[2] - roi_box[0]}, h={roi_box[3] - roi_box[1]}")
    if seed_points_file:
        print(f"Seed points: {len(seed_points)} loaded from {seed_points_file}")
    print(f"Number of detected objects: {len(rows)}")
    print(f"CSV: {output_dir / 'output_coordinates.csv'}")
    print(f"Mask: {output_dir / 'output_mask.tif'}")
    print(f"Original review image: {output_dir / 'original.png'}")
    print(f"Overlay: {output_dir / 'output_overlay.png'}")
    print(f"Numbered overlay: {output_dir / 'output_overlay_numbered.png'}")
    for row in rows:
        print(
            f"#{int(row['id']):02d}: "
            f"x={row['x_centroid_px']:.1f}, y={row['y_centroid_px']:.1f}, "
            f"area={row['area_px']:.0f} px"
        )


if __name__ == "__main__":
    # ========================
    # PARAMETERS TO EDIT HERE
    # ========================
    # Choose one mode: "absorption", "dark-field", or "phase-contrast".
    IMAGE_MODE = "absorption"

    # Suggested starting parameters for each image modality.
    # These are not universal constants: check output_overlay.png and tune if needed.
    MODE_CONFIGS = {
        "absorption": {
            "input_file": "image.tif",
            "output_folder": "outputs",
            "preprocess_mode": "none",
            "score_mode": "bright",
            "roi": (370, 260, 180, 178),
            "score_percentile": 97.2,
            "log_threshold": 0.11,
            "threshold_scale": 1.40,
            "ridge_weight": 0.0,
            "ignore_bottom_pixels": 0,
            "max_eccentricity": 0.88,
            "seed_points_file": "outputs/output_seed_points.csv",
            "absorption_intensity_file": "image.tif",
            "darkfield_intensity_file": "image.tif",
            "phase_intensity_file": "image.tif",
        },
        "dark-field": {
            "input_file": "image.tif",
            "output_folder": "outputs_darkfield",
            "preprocess_mode": "none",
            "score_mode": "bright",
            "roi": (370, 260, 180, 178),
            "score_percentile": 97,
            "log_threshold": 0.25,
            "threshold_scale": 1.0,
            "ridge_weight": 0.65,
            "ignore_bottom_pixels": 8,
            "max_eccentricity": 0.98,
            "seed_points_file": None,
            "absorption_intensity_file": "image.tif",
            "darkfield_intensity_file": "image.tif",
            "phase_intensity_file": "image.tif",
        },
        "phase-contrast": {
            "input_file": "image.tif",
            "output_folder": "outputs_phase",
            "preprocess_mode": "phase_contrast_enhanced",
            "score_mode": "phase_absolute",
            "roi": (410, 300, 130, 110),
            "score_percentile": 98.4,
            "log_threshold": 0.22,
            "threshold_scale": 1.0,
            "ridge_weight": 0.55,
            "ignore_bottom_pixels": 8,
            "max_eccentricity": 0.99,
            "seed_points_file": None,
            "absorption_intensity_file": "image.tif",
            "darkfield_intensity_file": "image.tif",
            "phase_intensity_file": "image.tif",
        },
    }

    MODE_ALIASES = {
        "ab": "absorption",
        "abs": "absorption",
        "absorption": "absorption",
        "sc": "dark-field",
        "darkfield": "dark-field",
        "dark-field": "dark-field",
        "df": "dark-field",
        "ph": "phase-contrast",
        "phase": "phase-contrast",
        "phasecontrast": "phase-contrast",
        "phase-contrast": "phase-contrast",
    }

    selected_mode = MODE_ALIASES.get(IMAGE_MODE.lower())
    if selected_mode is None:
        valid_modes = ", ".join(MODE_CONFIGS)
        raise ValueError(f"Unknown IMAGE_MODE={IMAGE_MODE!r}. Choose one of: {valid_modes}")

    config = MODE_CONFIGS[selected_mode]

    INPUT_FILE = config["input_file"]
    OUTPUT_FOLDER = config["output_folder"]
    PREPROCESS_MODE = config["preprocess_mode"]
    SCORE_MODE = config["score_mode"]
    SEED_POINTS_FILE = config["seed_points_file"]
    ABSORPTION_INTENSITY_FILE = config["absorption_intensity_file"]
    DARKFIELD_INTENSITY_FILE = config["darkfield_intensity_file"]
    PHASE_INTENSITY_FILE = config["phase_intensity_file"]

    # ROI = (x, y, width, height), where x = column and y = row.
    ROI = config["roi"]

    # Main thresholding method: "otsu", "yen", or "li".
    THRESHOLD_METHOD = "otsu"

    # Lower SCORE_PERCENTILE values make the detection more sensitive.
    # Example: 95 detects more weak regions, while 98 is stricter.
    SCORE_PERCENTILE = config["score_percentile"]

    # Lower LOG_THRESHOLD values make the LoG detection more sensitive.
    # Example: 0.10 detects more small blobs, while 0.18 is stricter.
    LOG_THRESHOLD = config["log_threshold"]
    THRESHOLD_SCALE = config["threshold_scale"]

    # Multi-scale top-hat catches small bright dots and irregular small clusters.
    TOPHAT_RADII = (2, 3, 4)

    # Ridge response catches weak elongated/filament-like target clusters.
    # Increase RIDGE_WEIGHT if weak regions are still missed.
    # Decrease it if elongated background artifacts become too visible.
    RIDGE_SIGMAS = (1.0, 1.5)
    RIDGE_WEIGHT = config["ridge_weight"]

    # Remove the bottom strip of the ROI if it overlaps a bright horizontal
    # support/background structure.
    IGNORE_BOTTOM_PIXELS = config["ignore_bottom_pixels"]

    # Geometric filters applied to segmented objects.
    MIN_AREA = 3
    MAX_AREA = 200
    MAX_ECCENTRICITY = config["max_eccentricity"]
    SEED_RADIUS = 5
    SEED_LOCAL_PERCENTILE = 80.0

    # Preprocessing parameters.
    SMOOTH_SIGMA = 0.7
    BACKGROUND_SIGMA = 7.0

    # Extra preprocessing for phase-contrast mode only.
    # Larger PHASE_BACKGROUND_SIGMA removes slower background variations.
    # CLAHE increases local contrast after background flattening.
    PHASE_BACKGROUND_SIGMA = 45.0
    PHASE_CLAHE_KERNEL_SIZE = 64
    PHASE_CLAHE_CLIP_LIMIT = 0.01

    print(f"Selected image mode: {selected_mode}")
    print(f"Preprocess mode: {PREPROCESS_MODE}")
    print(f"Score mode: {SCORE_MODE}")

    main(
        input_file=INPUT_FILE,
        output_folder=OUTPUT_FOLDER,
        roi=ROI,
        threshold_method=THRESHOLD_METHOD,
        score_percentile=SCORE_PERCENTILE,
        log_threshold=LOG_THRESHOLD,
        threshold_scale=THRESHOLD_SCALE,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        max_eccentricity=MAX_ECCENTRICITY,
        smooth_sigma=SMOOTH_SIGMA,
        background_sigma=BACKGROUND_SIGMA,
        tophat_radii=TOPHAT_RADII,
        ridge_sigmas=RIDGE_SIGMAS,
        ridge_weight=RIDGE_WEIGHT,
        ignore_bottom_pixels=IGNORE_BOTTOM_PIXELS,
        preprocess_mode=PREPROCESS_MODE,
        score_mode=SCORE_MODE,
        seed_points_file=SEED_POINTS_FILE,
        seed_radius=SEED_RADIUS,
        seed_local_percentile=SEED_LOCAL_PERCENTILE,
        absorption_intensity_file=ABSORPTION_INTENSITY_FILE,
        darkfield_intensity_file=DARKFIELD_INTENSITY_FILE,
        phase_intensity_file=PHASE_INTENSITY_FILE,
        phase_background_sigma=PHASE_BACKGROUND_SIGMA,
        phase_clahe_kernel_size=PHASE_CLAHE_KERNEL_SIZE,
        phase_clahe_clip_limit=PHASE_CLAHE_CLIP_LIMIT,
    )
