import cv2
import numpy as np
from typing import Tuple, Optional


def estimate_homography(
    pts1: np.ndarray,
    pts2: np.ndarray,
    method: int = cv2.RANSAC,
    ransac_reproj_threshold: float = 5.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate the homography matrix between two sets of matched points.
    
    Parameters
    ----------
    pts1 : np.ndarray
        Coordinates of matched points on Image 1, shape (N, 2)
    pts2 : np.ndarray
        Corresponding coordinates on Image 2, shape (N, 2)
    method : int, optional
        Method used for computing the homography matrix. 
        RANSAC is standard for robustness against outliers.
        default: cv2.RANSAC
    ransac_reproj_threshold : float, optional
        Maximum allowed reprojection error to treat a point pair as an inlier.
        default: 5.0
    
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        - M: 3x3 Homography matrix to transform pts1 to pts2. 
             Returns None if transformation cannot be found.
        - mask: N-element array where 1 indicates an inlier and 0 indicates an outlier.
                Returns None if transformation cannot be found.
    """

    # Need at least 4 points to compute the homography matrix
    if pts1 is None or pts2 is None or len(pts1) < 4 or len(pts2) < 4:
        print("Not enough points to estimate homography (minimum 4 required).")
        return None, None
    
    # Compute the homography matrix M: pts2 = M * pts1
    # Uses RANSAC algorithm to ignore bad feature matches (outliers)
    M, mask = cv2.findHomography(
        pts1, 
        pts2, 
        method, 
        ransac_reproj_threshold
    )
    
    # Sanity check for extreme scaling or perspective distortion
    if M is not None:
        det = np.linalg.det(M[0:2, 0:2])
        if det < 0.2 or det > 5.0 or abs(M[0,0]) < 0.5 or abs(M[1,1]) < 0.5:
            print("  [Homography] Extreme distortion detected. Falling back to Partial Affine transform.")
            M_affine, mask_affine = cv2.estimateAffinePartial2D(
                pts1, pts2, method=method, ransacReprojThreshold=ransac_reproj_threshold
            )
            if M_affine is not None:
                M = np.eye(3, dtype=np.float64)
                M[0:2, :] = M_affine
                mask = mask_affine
    
    return M, mask
