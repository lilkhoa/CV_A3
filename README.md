# CV - A3: Panoramic Photos – Image Stitching

This repository contains the implementation of an image stitching algorithm for creating panoramic photos. The algorithm consists of several steps, including feature detection, feature matching, homography estimation, and image blending.

## Usage

```bash
python main.py --folder <folder_path> --output <output_path>
```

### Example

```bash
python src/main.py --input "./data/BK-H6-imgs" --method SIFT --output "results/sift.jpg"
```