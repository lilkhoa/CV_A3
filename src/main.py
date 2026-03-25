import os
import argparse
import cv2
from pipeline import PanoramaStitcher

def main():
    parser = argparse.ArgumentParser(description="Image Stitching Pipeline (Panorama)")
    parser.add_argument("--input", type=str, required=True, help="Path to folder containing input images")
    parser.add_argument("--output", type=str, default="results/panorama.jpg", help="Path to save the output panorama")
    parser.add_argument("--method", type=str, choices=['SIFT', 'ORB', 'AKAZE'], default='SIFT', help="Feature extraction method")
    parser.add_argument("--features", type=int, default=3000, help="Number of features to extract")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold")
    
    args = parser.parse_args()
    
    # Initialize Stitcher
    stitcher = PanoramaStitcher(
        feature_method=args.method, 
        nfeatures=args.features, 
        match_ratio=args.ratio
    )
    
    # Run pipeline
    result = stitcher.stitch_folder(args.input, args.output)
    
    if result is not None:
        print(f"\nPanorama shape: {result.shape[1]}x{result.shape[0]}")
        print(f"Result saved to: {args.output}")

if __name__ == "__main__":
    main()
