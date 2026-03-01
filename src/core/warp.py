import cv2
import numpy as np


def warp_and_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    M: np.ndarray
) -> np.ndarray:
    """
    Warp Image 1 to align with Image 2 using the Homography matrix M,
    and blend them into a single panorama image.
    
    Parameters
    ----------
    img1 : np.ndarray
        The image to be warped (source image, e.g., right image)
    img2 : np.ndarray
        The reference image (destination image, e.g., left image)
    M : np.ndarray
        3x3 Homography matrix calculated to project img1 into img2's perspective
        
    Returns
    -------
    np.ndarray
        The resulting stitched image (panorama)
    """

    # Get dimensions of input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Extract corners of both images
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Predict the boundary of img1 after homography transformation
    pts1_transformed = cv2.perspectiveTransform(pts1, M)

    # Combine the corners of the reference image and the transformed image
    # to find the bounding box of the final panorama
    pts = np.concatenate((pts2, pts1_transformed), axis=0)

    # Find the minimum and maximum coordinates
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    # Calculate the translation shift needed to keep everything inside the positive coordinate range
    t = [-xmin, -ymin]
    
    # Create the translation matrix
    Ht = np.array([
        [1, 0, t[0]],
        [0, 1, t[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Warp img1 onto the final canvas size using the combined transformation (Translation * Homography)
    canvas_w = xmax - xmin
    canvas_h = ymax - ymin

    # Warp img1
    warped_img1 = cv2.warpPerspective(
        img1, 
        Ht.dot(M), 
        (canvas_w, canvas_h)
    )

    # Basic Blending: Overlay img2 onto the warped img1
    panorama = warped_img1.copy()
    
    # Overlay the reference image (img2) at the shifted position
    # The reference image is not distorted, just shifted by t[0], t[1]
    # Simple direct overwrite (basic blending)
    
    # Determine the region where img2 will be placed
    y_start = t[1]
    y_end = t[1] + h2
    x_start = t[0]
    x_end = t[0] + w2
    
    # Copy img2 pixels onto the panorama where img2 is present
    # To handle transparency/black backgrounds, consider the non-zero pixels
    mask = (img2 != 0)
    panorama[y_start:y_end, x_start:x_end][mask] = img2[mask]

    return panorama
