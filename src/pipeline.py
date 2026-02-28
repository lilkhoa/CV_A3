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

    def _create_distance_weight_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a weight mask using distance transform.
        Pixels further from the edge (black border) get higher weights.
        """
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        if dist.max() > 0:
            dist = dist / dist.max()
        return dist

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
        pairwise_H = []  # H[i] maps image i -> image i+1
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
            
            # H maps points in image i to image i+1's coordinate system
            H, mask = estimate_homography(pts1, pts2)
            
            if H is None:
                print(f"  [WARNING] Failed to compute homography between image {i+1} and {i+2}.")
                return None
            
            pairwise_H.append(H)

        # Step 4: Chain homographies relative to center image
        ref = n // 2  # Use middle image as reference (least distortion)
        print(f"\n[Step 4] Chaining homographies (reference image: {ref+1})...")
        
        # cumulative_H[i] = transformation from image i to the reference image's coordinate system
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

        # Step 6: Warp all images and blend using center-weighted blending
        print("\n[Step 6] Warping and blending all images...")
        
        panorama_final = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        max_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for i in range(n):
            # Combine translation with the cumulative homography
            H_final = Ht @ cumulative_H[i]
            
            # Warp the color image
            warped = cv2.warpPerspective(
                color_images[i],
                H_final,
                (canvas_w, canvas_h)
            )
            
            # Create weight mask based on valid pixels of warped image using Distance Transform
            mask_warped = cv2.warpPerspective(
                np.ones(color_images[i].shape[:2], dtype=np.uint8) * 255,
                H_final,
                (canvas_w, canvas_h)
            )
            
            warped_weight = self._create_distance_weight_mask(mask_warped)
            
            # Voronoi blending (sharp seam based on distance to center)
            win_mask = warped_weight > max_weight
            panorama_final[win_mask] = warped[win_mask]
            max_weight[win_mask] = warped_weight[win_mask]
            
            print(f"  Image {i+1} warped and blended.")

        panorama = panorama_final
        
        # Step 7: Crop black borders
        print("\n[Step 7] Cropping black borders...")
        panorama = self._crop_black_borders(panorama)

        print("\n--- Stitching Pipeline Completed ---")
        
        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, panorama)
            print(f"Final panorama saved to: {output_path}")
            
        return panorama

    def _crop_black_borders(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the black (zero) border regions from the panorama to get a clean rectangle.
        Iteratively shrinks the bounding box until no black pixels remain on its edges.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold: pixels > 0 are considered valid
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
        
        # Get the bounding box of the largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Iteratively shrink bounding box
        while w > 0 and h > 0:
            sub_img = thresh[y:y+h, x:x+w]
            if cv2.countNonZero(sub_img) == w * h:
                break
                
            # Find the edge with the most zeros and shrink it
            top = w - cv2.countNonZero(sub_img[0, :])
            bottom = w - cv2.countNonZero(sub_img[-1, :])
            left = h - cv2.countNonZero(sub_img[:, 0])
            right = h - cv2.countNonZero(sub_img[:, -1])
            
            m = max(top, bottom, left, right)
            if m == top: 
                y += 1
                h -= 1
            elif m == bottom: 
                h -= 1
            elif m == left: 
                x += 1
                w -= 1
            else: 
                w -= 1
        
        if w <= 0 or h <= 0:
            return img
            
        return img[y:y+h, x:x+w]
