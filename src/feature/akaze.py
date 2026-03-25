import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

def extract_akaze_features(
    gray_image: np.ndarray,
    nfeatures: int = 2000,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract AKAZE features from a single grayscale image.
    
    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale input image (obtained from preprocessing step)
    nfeatures : int, optional
        Maximum number of features to keep (default: 2000).
        Since AKAZE doesn't natively limit features on creation,
        we sort by response and keep the best ones.
    
    Returns
    -------
    Tuple[List[cv2.KeyPoint], np.ndarray]
        - keypoints: List of detected feature points with 2D coordinates (x, y),
                     size, and orientation angle
        - descriptors: Binary descriptor vectors (typically 61-bytes each for MLDB)
                      acting as "fingerprints" for keypoint matching
    """

    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = akaze.detectAndCompute(gray_image, None)
    
    # Handle empty detection
    if descriptors is None:
        return [], np.array([])
        
    # Limit the number of features to 'nfeatures' to match ORB/SIFT API consistency
    if len(keypoints) > nfeatures:
        # Sort keypoints based on their response (strength)
        pts_with_desc = list(zip(keypoints, descriptors))
        pts_with_desc.sort(key=lambda x: x[0].response, reverse=True)
        
        # Keep only the top 'nfeatures'
        pts_with_desc = pts_with_desc[:nfeatures]
        keypoints, descriptors = zip(*pts_with_desc)
        
        keypoints = list(keypoints)
        descriptors = np.array(descriptors)
        
    return keypoints, descriptors

def visualize_akaze_keypoints(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize detected AKAZE keypoints on an image.
    
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

    # Ensure image is in BGR format for drawing colors if it's grayscale
    if len(image.shape) == 2:
        draw_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        draw_image = image.copy()
    
    # Draw keypoints with size and orientation
    output_image = cv2.drawKeypoints(
        draw_image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    if output_path:
        cv2.imwrite(output_path, output_image)
        print(f"Keypoint visualization saved to {output_path}")
    
    return output_image