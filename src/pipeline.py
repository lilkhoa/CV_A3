import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

from core.preprocess import load_and_preprocess_images, get_image_list
from feature.sift import extract_sift_features
from feature.orb import extract_orb_features
from core.matcher import match_features
from core.homography import estimate_homography
from core.cylindrical import cylindrical_projection
from core.warp import warp_images_to_canvas
from core.blender import voronoi_blend, create_weight_mask
from utils import crop_black_borders


class PanoramaStitcher:
    """
    Image stitching pipeline using center-reference approach.
    
    Instead of sequentially warping onto a growing panorama (which accumulates
    distortion errors), this approach:
    1. Pre-computes pairwise homographies between consecutive ORIGINAL images.
    2. Chains them relative to a center reference image.
    3. Warps all images onto a single canvas in one step.
    4. Uses center-weighted blending for smooth transitions.
    """
    def __init__(self, feature_method: str = 'SIFT', nfeatures: int = 2000, match_ratio: float = 0.75, use_cylindrical: bool = True):
        self.feature_method = feature_method.upper()
        self.nfeatures = nfeatures
        self.match_ratio = match_ratio
        self.use_cylindrical = use_cylindrical

    def _extract_features(self, gray_image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        if self.feature_method == 'SIFT':
            return extract_sift_features(gray_image, nfeatures=self.nfeatures)
        elif self.feature_method == 'ORB':
            return extract_orb_features(gray_image, nfeatures=self.nfeatures)
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.feature_method}")

    def stitch_folder(self, image_folder: str, output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Run the full stitching pipeline on a folder of images.
        """
        print(f"--- Starting Stitching Pipeline using {self.feature_method} ---")
        
        # Step 1: Load and preprocess
        try:
            image_dict = load_and_preprocess_images(image_folder)
        except Exception as e:
            print(f"Error loading images: {e}")
            return None
            
        color_images = get_image_list(image_dict['color'])
        gray_images = get_image_list(image_dict['grayscale'])
        n = len(color_images)
        
        if n < 2:
            print("Need at least 2 images to stitch.")
            return None

        # Step 2: Extract features for ALL original images
        print("\n[Step 2] Projecting and Extracting features from all images...")
        all_kp = []
        all_desc = []
        
        for i in range(n):
            if self.use_cylindrical:
                # Approximate focal length as image width
                f = max(color_images[i].shape[0], color_images[i].shape[1])
                color_images[i] = cylindrical_projection(color_images[i], f)
                gray_images[i] = cylindrical_projection(gray_images[i], f)
            
            gray = gray_images[i]
            kp, desc = self._extract_features(gray)
            all_kp.append(kp)
            all_desc.append(desc)
            print(f"  Image {i+1}: {len(kp)} keypoints detected")

        # Step 3: Compute pairwise homographies between consecutive images
        print("\n[Step 3] Computing pairwise homographies...")
        pairwise_H = []
        for i in range(n - 1):
            print(f"  Matching image {i+1} <-> image {i+2}...")
            pts1, pts2 = match_features(
                all_kp[i], all_desc[i],
                all_kp[i+1], all_desc[i+1],
                method=self.feature_method,
                ratio=self.match_ratio
            )
            
            if len(pts1) < 4:
                print(f"  [WARNING] Not enough matches ({len(pts1)}) between image {i+1} and {i+2}.")
                return None
                
            print(f"  Found {len(pts1)} good matches.")
            
            H, mask = estimate_homography(pts1, pts2)
            
            if H is None:
                print(f"  [WARNING] Failed to compute homography between image {i+1} and {i+2}.")
                return None
            
            pairwise_H.append(H)

        # Step 4: Chain homographies relative to center image
        ref = n // 2
        print(f"\n[Step 4] Chaining homographies (reference image: {ref+1})...")
        
        cumulative_H = [None] * n
        cumulative_H[ref] = np.eye(3, dtype=np.float64)
        
        # Chain left: ref-1, ref-2, ..., 0
        # pairwise_H[i] maps image i -> image i+1
        # To go from image i to ref: cumulative_H[i] = cumulative_H[i+1] @ pairwise_H[i]
        for i in range(ref - 1, -1, -1):
            cumulative_H[i] = cumulative_H[i+1] @ pairwise_H[i]
        
        # Chain right: ref+1, ref+2, ..., n-1
        # pairwise_H[i-1] maps image i-1 -> image i
        # Inverse maps image i -> image i-1
        # cumulative_H[i] = cumulative_H[i-1] @ inv(pairwise_H[i-1])
        for i in range(ref + 1, n):
            H_inv = np.linalg.inv(pairwise_H[i - 1])
            cumulative_H[i] = cumulative_H[i - 1] @ H_inv

        # Step 5: Compute bounding box of the final panorama
        print("\n[Step 5] Computing canvas size...")
        all_corners = []
        for i in range(n):
            h, w = color_images[i].shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, cumulative_H[i])
            all_corners.append(transformed)

        all_corners = np.concatenate(all_corners, axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        canvas_w = int(xmax - xmin)
        canvas_h = int(ymax - ymin)
        print(f"  Canvas size: {canvas_w} x {canvas_h}")
        
        # Safety check
        MAX_CANVAS = 15000
        if canvas_w > MAX_CANVAS or canvas_h > MAX_CANVAS or canvas_w <= 0 or canvas_h <= 0:
            print(f"  [ERROR] Canvas too large ({canvas_w}x{canvas_h}). Aborting.")
            return None

        # Translation to shift everything into positive coordinates
        Ht = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ], dtype=np.float64)

        # Step 6: Warp all images onto the canvas
        print("\n[Step 6] Warping all images onto canvas...")
        
        # Prepare final homographies (translation + cumulative homography)
        final_homographies = [Ht @ cumulative_H[i] for i in range(n)]
        
        # Warp all images using the unified function
        warped_images = warp_images_to_canvas(
            color_images,
            final_homographies,
            (canvas_w, canvas_h)
        )
        
        # Create weight masks for Voronoi blending
        weight_masks = []
        for i in range(n):
            weight = create_weight_mask(
                color_images[i].shape[:2],
                final_homographies[i],
                (canvas_w, canvas_h)
            )
            weight_masks.append(weight)
            print(f"  Image {i+1} warped.")
        
        # Step 7: Blend images using Voronoi blending
        print("\n[Step 7] Blending images using Voronoi blending...")
        panorama = voronoi_blend(warped_images, weight_masks)
        print("  Blending complete.")

        # Step 8: Crop black borders
        print("\n[Step 8] Cropping black borders...")
        panorama = crop_black_borders(panorama)

        print("\n--- Stitching Pipeline Completed ---")
        
        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            base_dir = os.path.dirname(output_path)
            basename = os.path.basename(output_path)
            name, ext = os.path.splitext(basename)
            
            # 1. Save uncropped
            uncropped_path = os.path.join(base_dir, f"{name}_uncropped{ext}")
            cv2.imwrite(uncropped_path, panorama)
            print(f"Uncropped panorama saved to: {uncropped_path}")
            
            # 2. Save rect
            if bbox is not None:
                x, y, w, h = bbox
                rect_img = panorama.copy()
                thickness = max(2, int(max(panorama.shape[:2]) * 0.002))
                cv2.rectangle(rect_img, (x, y), (x+w, y+h), (0, 255, 0), thickness)
                rect_path = os.path.join(base_dir, f"{name}_rect{ext}")
                cv2.imwrite(rect_path, rect_img)
                print(f"Panorama with crop rect saved to: {rect_path}")
                
            # 3. Save cropped (final)
            cv2.imwrite(output_path, panorama_cropped)
            print(f"Final panorama saved to: {output_path}")
            
        return panorama
