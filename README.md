# Seed-Guided Grayscale Object Segmentation

This repository contains a Python workflow for segmenting small bright objects in grayscale `.tif` images. It combines manual seed selection with classical image processing methods such as local background subtraction, top-hat filtering, blob detection, connected components, and watershed splitting.

Raw images and generated results are intentionally not included.

## Repository Contents

```text
Select_Seed_Points.py
Segmentation_.py
requirements.txt
.gitignore
README.md
```

## Expected Input Structure

Place an input grayscale image in the following structure:

```text
project_folder/
└── Analyse/
    └── IMG.tif
```

The default scripts expect:

```text
Analyse/IMG.tif
```

You can change the file paths and ROI parameters directly in the parameter section near the bottom of each script.

## Workflow

1. Select seed points on the grayscale image:

```bash
python Select_Seed_Points.py
```

This creates a seed-point CSV file and preview images in:

```text
Analyse/Classical_methods/Segmentation_output/
```

2. Run the segmentation pipeline:

```bash
python Segmentation_.py
```

## Generated Outputs

The segmentation script generates:

```text
output_mask.tif
output_labels.tif
output_overlay.png
output_coordinates.csv
output_object_intensity_summary.csv
```

Additional debug and review outputs may also be generated depending on the parameter settings.

## Method Summary

The pipeline:

- loads a grayscale `.tif` image
- enhances small bright objects using local background subtraction and top-hat filters
- optionally uses manually selected seed points to guide segmentation
- separates touching regions using watershed
- filters segmented regions by size and shape
- exports masks, labels, overlays, coordinates, and intensity summaries

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Notes

This repository is designed as a code-only example. Image data, generated masks, overlays, CSV files, and other analysis outputs should remain outside version control unless they are explicitly safe to share.
