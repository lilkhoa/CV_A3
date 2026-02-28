import cv2
import numpy as np

def cylindrical_projection(img: np.ndarray, focal_length: float) -> np.ndarray:
    """
    Project an image onto a cylindrical surface.
    
    Parameters
    ----------
    img : np.ndarray
        The input image (BGR or grayscale).
    focal_length : float
        The estimated focal length of the camera.
        
    Returns
    -------
    np.ndarray
        The cylindrically projected image.
    """
    h, w = img.shape[:2]
    
    # Coordinate grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Center coordinates
    x_c = w / 2.0
    y_c = h / 2.0
    
    # Normalized coordinates
    x_norm = (x - x_c) / focal_length
    y_norm = (y - y_c) / focal_length
    
    # Cylindrical to Cartesian backward mapping
    # Determine corresponding angles and heights on the cylinder
    x_cyl = focal_length * np.arctan(x_norm)
    y_cyl = focal_length * y_norm / np.sqrt(x_norm**2 + 1)
    
    # Map back to image pixel coordinates
    map_x = focal_length * np.tan((x - x_c) / focal_length) + x_c
    map_y = (y - y_c) * np.sqrt((map_x - x_c)**2 / focal_length**2 + 1) + y_c
    
    # We use remap to apply the backward mapping
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    projected = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return projected
