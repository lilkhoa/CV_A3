import cv2
import numpy as np
from typing import List, Tuple, Optional

def alpha_blend(
    warped_images: List[np.ndarray],
    weight_masks: List[np.ndarray]
) -> np.ndarray:
    """
    Blend multiple warped images using alpha blending (weighted average).
    
    Parameters
    ----------
    warped_images : List[np.ndarray]
        List of warped images on the same canvas
    weight_masks : List[np.ndarray]
        Weight masks for each image. Each mask should be normalized [0, 1]
    
    Returns
    -------
    np.ndarray
        Blended panorama with smooth transitions
    """

    if not warped_images:
        raise ValueError("No images to blend")
    
    if not weight_masks:
        raise ValueError("Weight masks are required for alpha blending")
    
    canvas_h, canvas_w = warped_images[0].shape[:2]
    
    # Initialize accumulators
    blended = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    total_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    for warped, weight in zip(warped_images, weight_masks):
        if len(weight.shape) == 3:
            weight = cv2.cvtColor(weight, cv2.COLOR_BGR2GRAY)
        
        weight_3ch = np.expand_dims(weight, axis=2)
        blended += warped.astype(np.float32) * weight_3ch
        total_weight += weight
    
    mask = total_weight > 0
    for c in range(3):
        blended[:, :, c][mask] /= total_weight[mask]
    
    return blended.astype(np.uint8)

def voronoi_blend(
    warped_images: List[np.ndarray],
    weight_masks: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Blend multiple warped images using Voronoi blending (winner-takes-all).
    
    Parameters
    ----------
    warped_images : List[np.ndarray]
        List of warped images on the same canvas
    weight_masks : Optional[List[np.ndarray]], optional
        Pre-computed weight masks for each image. If None, will compute
        using distance transform. Each mask should be normalized [0, 1]
    
    Returns
    -------
    np.ndarray
        Blended panorama with sharp seams
    """
    if not warped_images:
        raise ValueError("No images to blend")
    
    canvas_h, canvas_w = warped_images[0].shape[:2]
    
    # Initialize output
    blended = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    max_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    # For each image, copy pixels where it has maximum weight
    for warped, weight in zip(warped_images, weight_masks):
        if len(weight.shape) == 3:
            weight = cv2.cvtColor(weight, cv2.COLOR_BGR2GRAY)
        
        win_mask = weight > max_weight
        blended[win_mask] = warped[win_mask]
        max_weight[win_mask] = weight[win_mask]
    
    return blended

def create_weight_mask(
    image_shape: Tuple[int, int],
    H: np.ndarray,
    canvas_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create a weight mask for an image after warping.
    
    Parameters
    ----------
    image_shape : Tuple[int, int]
        Original image shape as (height, width)
    H : np.ndarray
        Homography matrix
    canvas_size : Tuple[int, int]
        Canvas size as (width, height)
    
    Returns
    -------
    np.ndarray
        Normalized weight mask (0.0 to 1.0)
    """
    h, w = image_shape
    canvas_w, canvas_h = canvas_size
    
    # Create a binary mask for the original image
    mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask,
        H,
        (canvas_w, canvas_h)
    )
    
    dist = cv2.distanceTransform(warped_mask.astype(np.uint8), cv2.DIST_L2, 3)
    if dist.max() > 0:
        dist = dist / dist.max()
    
    return dist.astype(np.float32)
