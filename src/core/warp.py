import cv2
import numpy as np
from typing import Tuple, List, Optional

def warp_images_to_canvas(
    images: List[np.ndarray],
    homographies: List[np.ndarray],
    canvas_size: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Warp multiple images to a common canvas.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of input images
    homographies : List[np.ndarray]
        List of homography matrices (one per image)
    canvas_size : Tuple[int, int]
        Canvas size as (width, height)
    
    Returns
    -------
    List[np.ndarray]
        List of warped images
    """

    warped_images = []
    canvas_w, canvas_h = canvas_size
    
    for i, (image, H) in enumerate(zip(images, homographies)):
        warped = cv2.warpPerspective(
            image,
            H,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR
        )
        warped_images.append(warped)
    
    return warped_images
