import cv2
import numpy as np
from typing import List, Tuple, Optional


def alpha_blend_images(
    warped_images: List[np.ndarray],
    weight_masks: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Blend multiple warped images using alpha blending (feathering).
    
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
        Blended panorama image
    """
    if not warped_images:
        raise ValueError("No images to blend")
    
    canvas_h, canvas_w = warped_images[0].shape[:2]
    n_images = len(warped_images)
    
    # Generate weight masks if not provided
    if weight_masks is None:
        weight_masks = []
        for warped in warped_images:
            mask = _create_distance_weight_mask(warped)
            weight_masks.append(mask)
    
    # Initialize output
    blended = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    # Accumulate weighted images
    for i, (warped, weight) in enumerate(zip(warped_images, weight_masks)):
        # Ensure weight is 2D
        if len(weight.shape) == 3:
            weight = cv2.cvtColor(weight, cv2.COLOR_BGR2GRAY)
        
        # Expand weight to 3 channels for multiplication
        weight_3ch = np.stack([weight] * 3, axis=-1)
        
        # Add weighted image
        blended += warped.astype(np.float32) * weight_3ch
        weight_sum += weight
    
    # Normalize by total weight
    # Avoid division by zero
    weight_sum = np.maximum(weight_sum, 1e-10)
    weight_sum_3ch = np.stack([weight_sum] * 3, axis=-1)
    
    blended = blended / weight_sum_3ch
    
    # Convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
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
    
    # Warp the mask to the canvas
    warped_mask = cv2.warpPerspective(
        mask,
        H,
        (canvas_w, canvas_h)
    )
    
    # Apply distance transform
    dist = cv2.distanceTransform(warped_mask.astype(np.uint8), cv2.DIST_L2, 3)
    
    # Normalize to [0, 1]
    if dist.max() > 0:
        dist = dist / dist.max()
    
    return dist.astype(np.float32)


def _create_distance_weight_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a weight mask using distance transform.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (warped, may have black borders)
    
    Returns
    -------
    np.ndarray
        Normalized weight mask [0, 1]
    """
    # Create binary mask of valid pixels
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    mask = (gray > 0).astype(np.uint8) * 255
    
    # Apply distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    
    # Normalize to [0, 1]
    if dist.max() > 0:
        dist = dist / dist.max()
    
    return dist.astype(np.float32)
