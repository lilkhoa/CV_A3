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

def multiband_blend(
    warped_images: List[np.ndarray],
    weight_masks: List[np.ndarray],
    levels: int = 6
) -> np.ndarray:
    """
    Blend multiple warped images using multiband (Laplacian pyramid) blending.
 
    Parameters
    ----------
    warped_images : List[np.ndarray]
        List of warped images on the same canvas
    weight_masks : List[np.ndarray]
        Weight masks for each image. Each mask should be normalized [0, 1]
    levels : int, optional
        Pyramid depth. Default is 6
 
    Returns
    -------
    np.ndarray
        Blended panorama with smooth transitions and no ghosting
    """
    if not warped_images:
        raise ValueError("No images to blend")
 
    if len(warped_images) != len(weight_masks):
        raise ValueError("warped_images and weight_masks must have the same length")
 
    canvas_h, canvas_w = warped_images[0].shape[:2]
 
    min_dim = min(canvas_h, canvas_w)
    max_levels = int(np.floor(np.log2(min_dim))) - 1
    levels = min(levels, max_levels)
 
    weight_stack = np.stack(weight_masks, axis=0)
 
    for i, img in enumerate(warped_images):
        valid = np.any(img > 0, axis=2).astype(np.float32)
        weight_stack[i] *= valid
 
    total = weight_stack.sum(axis=0, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    norm_weights = weight_stack / total
 
    blended_pyramid = None
 
    for img, w_map in zip(warped_images, norm_weights):
        w_3ch = np.stack([w_map] * 3, axis=2)
 
        lap_img = _build_laplacian_pyramid(img.astype(np.float32), levels)
        lap_w = _build_gaussian_pyramid(w_3ch, levels)
 
        if blended_pyramid is None:
            blended_pyramid = [np.zeros_like(l) for l in lap_img]
 
        for lvl in range(levels):
            blended_pyramid[lvl] += lap_img[lvl] * lap_w[lvl]
 
    result = _reconstruct_from_laplacian(blended_pyramid)
    return np.clip(result, 0, 255).astype(np.uint8)

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
 
def _build_gaussian_pyramid(
    img: np.ndarray,
    levels: int
) -> List[np.ndarray]:
    pyramid = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid
 
 
def _build_laplacian_pyramid(
    img: np.ndarray,
    levels: int
) -> List[np.ndarray]:
    gauss = _build_gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels - 1):
        up = cv2.pyrUp(gauss[i + 1], dstsize=(gauss[i].shape[1], gauss[i].shape[0]))
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])
    return lap
 
 
def _reconstruct_from_laplacian(
    pyramid: List[np.ndarray]
) -> np.ndarray:
    result = pyramid[-1]
    for lap in reversed(pyramid[:-1]):
        result = cv2.pyrUp(result, dstsize=(lap.shape[1], lap.shape[0])) + lap
    return result