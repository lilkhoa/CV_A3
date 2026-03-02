import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

from feature.sift import visualize_sift_keypoints
from feature.orb import visualize_orb_keypoints
from core.matcher import visualize_matches

from pipeline import PanoramaStitcher
from core.preprocess import load_and_preprocess_images, get_image_list
from core.matcher import match_features

def visualize_keypoints(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    method: str = 'SIFT',
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Unified interface to visualize keypoints using the appropriate method.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color)
    keypoints : List[cv2.KeyPoint]
        Detected keypoints to visualize
    method : str, optional
        Feature extraction method: 'SIFT' or 'ORB' (default: 'SIFT')
    output_path : Optional[str]
        If provided, save the visualization to this path
    
    Returns
    -------
    np.ndarray
        Image with keypoints drawn
    """
    method = method.upper()
    
    if method == 'SIFT':
        return visualize_sift_keypoints(image, keypoints, output_path)
    elif method == 'ORB':
        return visualize_orb_keypoints(image, keypoints, output_path)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'SIFT' or 'ORB'.")


def visualize_all_keypoints(
    images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    method: str = 'SIFT',
    output_dir: Optional[str] = None,
    prefix: str = 'keypoints'
) -> List[np.ndarray]:
    """
    Visualize keypoints for multiple images.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of input images
    all_keypoints : List[List[cv2.KeyPoint]]
        List of keypoint lists, one for each image
    method : str, optional
        Feature extraction method: 'SIFT' or 'ORB' (default: 'SIFT')
    output_dir : Optional[str]
        If provided, save visualizations to this directory
    prefix : str, optional
        Prefix for output filenames (default: 'keypoints')
    
    Returns
    -------
    List[np.ndarray]
        List of visualization images
    """
    if len(images) != len(all_keypoints):
        raise ValueError("Number of images and keypoint lists must match")
    
    visualizations = []
    
    for i, (img, kp) in enumerate(zip(images, all_keypoints)):
        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{prefix}_{i+1:02d}.jpg")
        
        vis_img = visualize_keypoints(img, kp, method, output_path)
        visualizations.append(vis_img)
        
        print(f"  Image {i+1}: {len(kp)} keypoints visualized")
    
    return visualizations


def create_keypoints_grid(
    images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    method: str = 'SIFT',
    output_path: Optional[str] = None,
    max_cols: int = 3
) -> np.ndarray:
    """
    Create a grid visualization of all images with their keypoints.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of input images
    all_keypoints : List[List[cv2.KeyPoint]]
        List of keypoint lists, one for each image
    method : str, optional
        Feature extraction method: 'SIFT' or 'ORB' (default: 'SIFT')
    output_path : Optional[str]
        If provided, save the grid to this path
    max_cols : int, optional
        Maximum number of columns in the grid (default: 3)
    
    Returns
    -------
    np.ndarray
        Grid visualization image
    """
    n = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    
    # Visualize keypoints on all images
    vis_images = []
    for img, kp in zip(images, all_keypoints):
        vis_img = visualize_keypoints(img, kp, method)
        vis_images.append(vis_img)
    
    # Resize all images to the same size for grid
    max_h = max(img.shape[0] for img in vis_images)
    max_w = max(img.shape[1] for img in vis_images)
    target_h = 400  # Fixed height for grid cells
    target_w = int(target_h * max_w / max_h)
    
    resized_images = []
    for img in vis_images:
        resized = cv2.resize(img, (target_w, target_h))
        resized_images.append(resized)
    
    # Fill remaining cells with blank images
    while len(resized_images) < rows * cols:
        blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized_images.append(blank)
    
    # Create grid
    grid_rows = []
    for r in range(rows):
        row_images = resized_images[r * cols:(r + 1) * cols]
        grid_row = np.hstack(row_images)
        grid_rows.append(grid_row)
    
    grid = np.vstack(grid_rows)
    
    if output_path:
        cv2.imwrite(output_path, grid)
        print(f"Keypoints grid saved to {output_path}")
    
    return grid


def visualize_pairwise_matches(
    images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    all_matches: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: Optional[str] = None,
    prefix: str = 'matches',
    max_matches: int = 50
) -> List[np.ndarray]:
    """
    Visualize pairwise matches for consecutive image pairs.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of input images
    all_keypoints : List[List[cv2.KeyPoint]]
        List of keypoint lists, one for each image
    all_matches : List[Tuple[np.ndarray, np.ndarray]]
        List of (pts1, pts2) tuples for each consecutive pair
    output_dir : Optional[str]
        If provided, save visualizations to this directory
    prefix : str, optional
        Prefix for output filenames (default: 'matches')
    max_matches : int, optional
        Maximum number of matches to display per pair (default: 50)
    
    Returns
    -------
    List[np.ndarray]
        List of match visualization images
    """
    if len(images) - 1 != len(all_matches):
        raise ValueError("Number of match pairs should be len(images) - 1")
    
    visualizations = []
    
    for i, (pts1, pts2) in enumerate(all_matches):
        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{prefix}_{i+1:02d}_{i+2:02d}.jpg")
        
        vis_img = visualize_matches(
            images[i], images[i + 1],
            all_keypoints[i], all_keypoints[i + 1],
            pts1, pts2,
            output_path,
            max_matches
        )
        visualizations.append(vis_img)
        
        print(f"  Match visualization {i+1} <-> {i+2}: {len(pts1)} matches")
    
    return visualizations


def create_matches_comparison(
    images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    all_matches: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[str] = None,
    max_matches: int = 30
) -> np.ndarray:
    """
    Create a vertical stacked comparison of all pairwise matches.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of input images
    all_keypoints : List[List[cv2.KeyPoint]]
        List of keypoint lists, one for each image
    all_matches : List[Tuple[np.ndarray, np.ndarray]]
        List of (pts1, pts2) tuples for each consecutive pair
    output_path : Optional[str]
        If provided, save the comparison to this path
    max_matches : int, optional
        Maximum number of matches to display per pair (default: 30)
    
    Returns
    -------
    np.ndarray
        Stacked comparison image
    """
    match_vis = []
    
    for i, (pts1, pts2) in enumerate(all_matches):
        vis_img = visualize_matches(
            images[i], images[i + 1],
            all_keypoints[i], all_keypoints[i + 1],
            pts1, pts2,
            None,
            max_matches
        )
        
        # Resize to standard width
        target_w = 1200
        h, w = vis_img.shape[:2]
        target_h = int(h * target_w / w)
        resized = cv2.resize(vis_img, (target_w, target_h))
        
        # Add label
        label = f"Match {i+1} <-> {i+2} ({len(pts1)} matches)"
        cv2.putText(
            resized, label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 0), 2, cv2.LINE_AA
        )
        
        match_vis.append(resized)
    
    # Stack vertically
    comparison = np.vstack(match_vis)
    
    if output_path:
        cv2.imwrite(output_path, comparison)
        print(f"Matches comparison saved to {output_path}")
    
    return comparison


def create_feature_summary(
    images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    all_matches: List[Tuple[np.ndarray, np.ndarray]],
    method: str = 'SIFT',
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a comprehensive summary visualization with statistics.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of input images
    all_keypoints : List[List[cv2.KeyPoint]]
        List of keypoint lists, one for each image
    all_matches : List[Tuple[np.ndarray, np.ndarray]]
        List of (pts1, pts2) tuples for each consecutive pair
    method : str, optional
        Feature extraction method: 'SIFT' or 'ORB' (default: 'SIFT')
    output_path : Optional[str]
        If provided, save the summary to this path
    
    Returns
    -------
    np.ndarray
        Summary visualization image
    """
    # Create statistics text image
    n_images = len(images)
    stats = [
        f"Feature Extraction Summary ({method})",
        f"=" * 50,
        f"Total Images: {n_images}",
        f"",
        f"Keypoints per image:",
    ]
    
    for i, kp in enumerate(all_keypoints):
        stats.append(f"  Image {i+1}: {len(kp)} keypoints")
    
    stats.append(f"")
    stats.append(f"Matches per consecutive pair:")
    
    for i, (pts1, pts2) in enumerate(all_matches):
        stats.append(f"  Image {i+1} <-> {i+2}: {len(pts1)} matches")
    
    # Calculate statistics
    total_keypoints = sum(len(kp) for kp in all_keypoints)
    avg_keypoints = total_keypoints / n_images
    total_matches = sum(len(pts1) for pts1, _ in all_matches)
    avg_matches = total_matches / len(all_matches) if all_matches else 0
    
    stats.extend([
        f"",
        f"Statistics:",
        f"  Total keypoints: {total_keypoints}",
        f"  Average keypoints per image: {avg_keypoints:.1f}",
        f"  Total matches: {total_matches}",
        f"  Average matches per pair: {avg_matches:.1f}",
    ])
    
    # Create text image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 30
    margin = 20
    
    text_img_height = len(stats) * line_height + 2 * margin
    text_img_width = 600
    text_img = np.ones((text_img_height, text_img_width, 3), dtype=np.uint8) * 255
    
    for i, line in enumerate(stats):
        y = margin + (i + 1) * line_height
        cv2.putText(
            text_img, line,
            (margin, y),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )
    
    # Create keypoints grid (small version)
    grid = create_keypoints_grid(images, all_keypoints, method, None, max_cols=3)
    
    # Resize grid to match text width
    h, w = grid.shape[:2]
    target_h = int(h * text_img_width / w)
    grid_resized = cv2.resize(grid, (text_img_width, target_h))
    
    # Combine text and grid
    summary = np.vstack([text_img, grid_resized])
    
    if output_path:
        cv2.imwrite(output_path, summary)
        print(f"Feature summary saved to {output_path}")
    
    return summary


def save_all_visualizations(
    image_folder: str,
    images: List[np.ndarray],
    all_keypoints: List[List[cv2.KeyPoint]],
    all_matches: List[Tuple[np.ndarray, np.ndarray]],
    method: str = 'SIFT',
    output_dir: str = 'results/visualizations'
) -> None:
    """
    Generate and save all visualization types to a directory.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of input images
    all_keypoints : List[List[cv2.KeyPoint]]
        List of keypoint lists, one for each image
    all_matches : List[Tuple[np.ndarray, np.ndarray]]
        List of (pts1, pts2) tuples for each consecutive pair
    method : str, optional
        Feature extraction method: 'SIFT' or 'ORB' (default: 'SIFT')
    output_dir : str, optional
        Directory to save all visualizations (default: 'results/visualizations')
    """
    output_dir = os.path.join(output_dir, os.path.basename(image_folder))
    output_dir = os.path.join(output_dir, method.lower())

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations in {output_dir}...")
    
    # 1. Individual keypoint visualizations
    print("\n[1/5] Visualizing keypoints for each image...")
    keypoints_dir = os.path.join(output_dir, 'keypoints')
    visualize_all_keypoints(images, all_keypoints, method, keypoints_dir)
    
    # 2. Keypoints grid
    print("\n[2/5] Creating keypoints grid...")
    grid_path = os.path.join(output_dir, 'keypoints_grid.jpg')
    create_keypoints_grid(images, all_keypoints, method, grid_path)
    
    # 3. Pairwise match visualizations
    print("\n[3/5] Visualizing pairwise matches...")
    matches_dir = os.path.join(output_dir, 'matches')
    visualize_pairwise_matches(images, all_keypoints, all_matches, matches_dir)
    
    # 4. Matches comparison
    print("\n[4/5] Creating matches comparison...")
    comparison_path = os.path.join(output_dir, 'matches_comparison.jpg')
    create_matches_comparison(images, all_keypoints, all_matches, comparison_path)
    
    # 5. Feature summary
    print("\n[5/5] Creating feature summary...")
    summary_path = os.path.join(output_dir, 'feature_summary.jpg')
    create_feature_summary(images, all_keypoints, all_matches, method, summary_path)
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":

    image_folder = os.path.join('data', 'BK-H6-imgs')
    image_dict = load_and_preprocess_images(image_folder)
    color_images = get_image_list(image_dict['color'])
    gray_images = get_image_list(image_dict['grayscale'])
    
    stitcher = PanoramaStitcher(feature_method='ORB', nfeatures=2000, match_ratio=0.75)
    
    all_kp = []
    all_desc = []
    
    for gray in gray_images:
        kp, desc = stitcher._extract_features(gray)
        all_kp.append(kp)
        all_desc.append(desc)
    
    all_matches = []
    for i in range(len(gray_images) - 1):
        pts1, pts2 = match_features(
            all_kp[i], all_desc[i],
            all_kp[i+1], all_desc[i+1],
            method=stitcher.feature_method,
            ratio=stitcher.match_ratio
        )
        all_matches.append((pts1, pts2))
    
    save_all_visualizations(image_folder, color_images, all_kp, all_matches, stitcher.feature_method)
