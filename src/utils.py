import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

def crop_black_borders(img: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Crop the black (zero) border regions using the Maximum Inscribed Rectangle algorithm.
        To ensure speed, it downscales the mask, finds the approximate max rectangle, 
        and then refines it at full resolution.
        Returns the cropped image and the bounding box.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        
        # 1. Downscale for fast approximation
        # Use INTER_AREA so a downscaled pixel is only 255 if the ENTIRE block is 255
        scale = 0.25
        small_mask = cv2.resize(thresh, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, small_mask = cv2.threshold(small_mask, 254, 255, cv2.THRESH_BINARY)
        h, w = small_mask.shape
        
        # 2. Find max inscribed rectangle on small mask
        max_area = 0
        best_rect = (0, 0, 0, 0)
        heights = np.zeros(w, dtype=np.int32)
        
        for i in range(h):
            heights = np.where(small_mask[i] > 0, heights + 1, 0)
            stack = []
            for j in range(w + 1):
                curr_h = heights[j] if j < w else 0
                start_j = j
                while stack and stack[-1][1] > curr_h:
                    pos, height = stack.pop()
                    area = height * (j - pos)
                    if area > max_area:
                        max_area = area
                        best_rect = (pos, i - height + 1, j - pos, height)
                    start_j = pos
                stack.append((start_j, curr_h))
                
        # 3. Scale back to original resolution
        x, y, w_r, h_r = best_rect
        x = int(x / scale)
        y = int(y / scale)
        w_r = int(w_r / scale)
        h_r = int(h_r / scale)
        
        # Ensure within bounds
        h_full, w_full = thresh.shape
        x = max(0, x)
        y = max(0, y)
        w_r = min(w_full - x, w_r)
        h_r = min(h_full - y, h_r)
        
        # 4. Refine at full resolution (shrink if any black pixels remain at edges)
        while w_r > 0 and h_r > 0:
            sub_img = thresh[y:y+h_r, x:x+w_r]
            if cv2.countNonZero(sub_img) == w_r * h_r:
                break
                
            top = w_r - cv2.countNonZero(sub_img[0, :])
            bottom = w_r - cv2.countNonZero(sub_img[-1, :])
            left = h_r - cv2.countNonZero(sub_img[:, 0])
            right = h_r - cv2.countNonZero(sub_img[:, -1])
            
            m = max(top, bottom, left, right)
            if m == top: 
                y += 1
                h_r -= 1
            elif m == bottom: 
                h_r -= 1
            elif m == left: 
                x += 1
                w_r -= 1
            else: 
                w_r -= 1
                
        if w_r <= 0 or h_r <= 0:
            return img, None
            
        return img[y:y+h_r, x:x+w_r], (x, y, w_r, h_r)