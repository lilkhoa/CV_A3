import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


def extract_orb_features(
    gray_image: np.ndarray,
    nfeatures: int = 2000
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract ORB features from a single grayscale image.
    
    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale input image (obtained from preprocessing step)
    nfeatures : int, optional
        Maximum number of features to detect (default: 2000)
        Limiting features prevents slow matching on high-detail images
    
    Returns
    -------
    Tuple[List[cv2.KeyPoint], np.ndarray]
        - keypoints: List of detected feature points with 2D coordinates (x, y),
                     size, and orientation angle
        - descriptors: Binary descriptor vectors (256 bits each)
                      acting as "fingerprints" for keypoint matching             
    """

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def visualize_orb_keypoints(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize detected ORB keypoints on an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (can be grayscale or color)
    keypoints : List[cv2.KeyPoint]
        Detected keypoints to visualize
    output_path : Optional[str]
        If provided, save the visualization to this path
    
    Returns
    -------
    np.ndarray
        Image with keypoints drawn (circles and orientation lines)
    """

    # Draw keypoints with size and orientation
    output_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    if output_path:
        cv2.imwrite(output_path, output_image)
        print(f"Keypoint visualization saved to {output_path}")
    
    return output_image
