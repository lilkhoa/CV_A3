# Panoramic Image Stitching

A robust and flexible image stitching pipeline for creating high-quality panoramic photos from overlapping image sequences.

[![Python 3.11+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)

---

## Overview

This project implements a comprehensive image stitching algorithm using a **center-reference approach** that avoids cumulative distortion errors inherent in sequential stitching methods. The pipeline handles multiple stages of image processing: feature detection, feature matching, homography estimation, image warping, and intelligent blending.

### Key Features

- **Multiple Feature Extraction Methods**: Support for SIFT, ORB, and AKAZE feature descriptors
- **Flexible Feature Matching**: Lowe's ratio test with configurable thresholds for robust matching
- **Center-Reference Stitching**: Pre-computes homographies relative to a center image to minimize distortion
- **Cylindrical Projection**: Optional cylindrical warping for better wide-angle results
- **Advanced Blending Techniques**:
  - Voronoi blending (winner-takes-all for sharp seams)
  - Alpha blending (smooth weighted transitions)
  - Multiband blending (pyramid-based blending for seamless transitions)
- **Comprehensive Visualization**: Tools to visualize feature extraction and matching results
- **Automatic Border Cropping**: Intelligent removal of black borders from the final panorama
- **Robust Error Handling**: Graceful handling of edge cases and invalid images

---

## Prerequisites

- **Python**: 3.11 or higher
- **Operating System**: Windows, macOS, or Linux

---

## Installation

### 1. Clone the Repository

```bash
git https://github.com/lilkhoa/CV_A3.git
cd CV_A3
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependency is:
- **opencv-python**: Computer vision library for image processing and feature detection

---

## Usage

### Basic Image Stitching

Run the stitching pipeline on a folder of images:

```bash
python src/main.py --input <image_folder> --output <output_filename>
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | *required* | Path to folder containing input images |
| `--output` | str | `panorama.jpg` | Output filename (saved in `results/` folder) |
| `--method` | str | `SIFT` | Feature extraction method: `SIFT`, `ORB`, or `AKAZE` |
| `--features` | int | `3000` | Number of features to extract per image |
| `--ratio` | float | `0.75` | Lowe's ratio test threshold for feature matching |

### Examples

#### Basic Usage with SIFT

```bash
python src/main.py --input "./data/BK-H6-imgs" --output "result.jpg"
```

This will:
1. Load all images from `./data/BK-H6-imgs`
2. Extract 3000 SIFT keypoints per image
3. Match features between consecutive images
4. Compute homographies and chain them relative to the center image
5. Warp all images onto a unified canvas
6. Blend overlapping regions
7. Crop black borders automatically
8. Save results in `results/` folder

#### Using ORB Features (Faster, Less Memory)

```bash
python src/main.py --input "./data/images" --output "orb_result.jpg" --method ORB --features 2000
```

#### Fine-tuning Matching Quality

```bash
python src/main.py --input "./data/images" --output "high_quality.jpg" --ratio 0.7 --features 5000
```

Lower ratio values (e.g., 0.7) result in more stringent matching, reducing outliers.

### Output Files

The stitching pipeline generates multiple output files in the `results/` folder:

```
results/
├── result_uncropped.jpg      # Full panorama with black borders
├── result_rect.jpg           # Panorama showing crop rectangle
└── result.jpg                # Final cropped panorama (main result)
```

---

## Advanced Usage

### Using Different Blending Methods

The pipeline uses Multiband blending by default. You can modify the blending strategy in the pipeline by changing the blending function in `pipeline.py`:
```python
# Example: Switch to Voronoi blending
from core.blender import voronoi_blend
# In the blending step
panorama = voronoi_blend(warped_images, weight_masks)
```

