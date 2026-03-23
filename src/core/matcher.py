import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


def match_features(
    kp1: List[cv2.KeyPoint],
    desc1: np.ndarray,
    kp2: List[cv2.KeyPoint],
    desc2: np.ndarray,
    method: str = 'ORB',
    ratio: float = 0.75
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match features between two images and filter outliers.
    
    Parameters
    ----------
    kp1 : List[cv2.KeyPoint]
        Keypoints from Image 1
    desc1 : np.ndarray
        Descriptors from Image 1
    kp2 : List[cv2.KeyPoint]
        Keypoints from Image 2
    desc2 : np.ndarray
        Descriptors from Image 2
    method : str, optional
        Feature extraction method: 'ORB' or 'SIFT' (default: 'ORB')
    ratio : float, optional
        Lowe's ratio test threshold (default: 0.75)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - pts1: Coordinates of matched points on Image 1, shape (N, 2)
                Format: [[x1, y1], [x2, y2], ...]
        - pts2: Corresponding coordinates on Image 2, shape (N, 2)
                Format: [[x1, y1], [x2, y2], ...]
        Both arrays have the same length and correspond by index.
    """

    # Handle edge cases
    if desc1 is None or desc2 is None:
        return np.array([]), np.array([])
    if len(desc1) < 2 or len(desc2) < 2:
        return np.array([]), np.array([])
    
    # Step 1: Initialize the Matcher based on method
    # Brute-Force matcher is simple. 
    # It takes the descriptor of one feature in first set and is matched with 
    # all other features in second set using some distance calculation. And the closest one is returned.
    if method.upper() in ['ORB', 'AKAZE']:
        # ORB and AKAZE use binary descriptors -> Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == 'SIFT':
        # SIFT uses floating-point descriptors -> Euclidean distance
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'ORB' or 'SIFT'.")
    
    # Step 2: k-Nearest Neighbors matching (k-NN)
    # k=2 -> we want the two best matches for each descriptor.
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Step 3: Noise filtering using Lowe's Ratio Test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            
            # Lowe's Ratio Test: Is the best match significantly better?
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    
    # Step 4: Extract coordinates (x, y)
    if len(good_matches) == 0:
        return np.array([]), np.array([])
    
    pts1 = []
    pts2 = []
    
    for m in good_matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    return pts1, pts2


def visualize_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    pts1: np.ndarray,
    pts2: np.ndarray,
    output_path: Optional[str] = None,
    max_matches: int = 50
) -> np.ndarray:
    """
    Visualize matched keypoints between two images.
    
    Parameters
    ----------
    img1 : np.ndarray
        First image
    img2 : np.ndarray
        Second image
    kp1 : List[cv2.KeyPoint]
        Keypoints from first image
    kp2 : List[cv2.KeyPoint]
        Keypoints from second image
    pts1 : np.ndarray
        Matched coordinates from first image
    pts2 : np.ndarray
        Matched coordinates from second image
    output_path : Optional[str]
        If provided, save visualization to this path
    max_matches : int, optional
        Maximum number of matches to display (default: 50)
    
    Returns
    -------
    np.ndarray
        Visualization image with matches drawn
    """

    matches_to_draw = []
    num_to_draw = min(len(pts1), max_matches)
    
    for i in range(num_to_draw):
        pt1 = tuple(pts1[i])
        pt2 = tuple(pts2[i])
        
        idx1 = next((idx for idx, kp in enumerate(kp1) 
                    if np.allclose(kp.pt, pt1, atol=0.1)), None)
        idx2 = next((idx for idx, kp in enumerate(kp2) 
                    if np.allclose(kp.pt, pt2, atol=0.1)), None)
        
        if idx1 is not None and idx2 is not None:
            match = cv2.DMatch(idx1, idx2, 0)
            matches_to_draw.append(match)
    
    # Draw matches
    output_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, output_img)
        print(f"Match visualization saved to {output_path}")
    
    return output_img
