import cv2
import numpy as np


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    """
    Crop black border regions from a panorama image.
    
    Iteratively shrinks the bounding box until no black pixels remain on edges.
    
    Parameters
    ----------
    image : np.ndarray
        Input panorama image
    
    Returns
    -------
    np.ndarray
        Cropped image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold: pixels > 0 are valid
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get bounding box of largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Iteratively shrink bounding box
    while w > 0 and h > 0:
        sub_img = thresh[y:y+h, x:x+w]
        if cv2.countNonZero(sub_img) == w * h:
            break
        
        # Find edge with most zeros and shrink it
        top = w - cv2.countNonZero(sub_img[0, :])
        bottom = w - cv2.countNonZero(sub_img[-1, :])
        left = h - cv2.countNonZero(sub_img[:, 0])
        right = h - cv2.countNonZero(sub_img[:, -1])
        
        m = max(top, bottom, left, right)
        if m == top:
            y += 1
            h -= 1
        elif m == bottom:
            h -= 1
        elif m == left:
            x += 1
            w -= 1
        else:
            w -= 1
    
    if w <= 0 or h <= 0:
        return image
    
    return image[y:y+h, x:x+w]
