import cv2
import numpy as np
import os
from typing import Dict, Tuple, List


def load_and_preprocess_images(
    image_folder: str,
    max_width: int = 800
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load and preprocess images from a folder for panorama stitching.
    
    Parameters
    ----------
    image_folder : str
        Path to the folder containing input images
    max_width : int, optional
        Maximum width for resizing images (default: 800)
        Images wider than this will be resized proportionally
    
    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A dictionary containing two sets of images:
        {
            'grayscale': {
                'filename1': gray_image1,
                'filename2': gray_image2,
                ...
            },
            'color': {
                'filename1': color_image1,
                'filename2': color_image2,
                ...
            }
        }
        - Grayscale images: For feature extraction (ORB/SIFT)
        - Color images: For image alignment and final panorama stitching
    """

    # Initialize storage for both image sets
    grayscale_images = {}
    color_images = {}
    
    # Get all image files from the folder
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(supported_extensions)
    ]
    image_files.sort()
    if not image_files:
        raise ValueError(f"No images found in folder: {image_folder}")
    print(f"Found {len(image_files)} images in {image_folder}")
    
    for filename in image_files:
        # Step 1: Read Image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {filename}, skipping...")
            continue
        print(f"Processing {filename}: Original size {image.shape[1]}x{image.shape[0]}")
        
        # Step 2: Resize image
        height, width = image.shape[:2]
        if width > max_width:
            scale_factor = max_width / width
            new_width = max_width
            new_height = int(height * scale_factor)
            
            # Resize the image
            resized_image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            print(f"  -> Resized to {new_width}x{new_height} (scale: {scale_factor:.3f})")
        else: # No resizing
            resized_image = image.copy()
            print(f"  -> No resize needed")
        
        # Step 3: Grayscale Conversion
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        grayscale_images[filename] = gray
        color_images[filename] = resized_image
    
    result = {
        'grayscale': grayscale_images,
        'color': color_images
    }
    return result


def get_image_list(images_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Convert a dictionary of images to a sorted list.
    
    Parameters
    ----------
    images_dict : Dict[str, np.ndarray]
        Dictionary mapping filenames to images
    
    Returns
    -------
    List[np.ndarray]
        List of images in sorted filename order
    """
    sorted_filenames = sorted(images_dict.keys())
    return [images_dict[filename] for filename in sorted_filenames]
