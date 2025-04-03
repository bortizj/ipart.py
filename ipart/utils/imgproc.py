"""
Copyleft 2025
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

author: Benhur Ortiz-Jaramillo
"""

import numpy as np
import cv2


def check_and_adjust_image_size(in_bgr: np.ndarray, tgt_size: tuple[int, int] = (180, 320)) -> np.ndarray:
    """
    Making images lower than a standard size for computational purposes
    """
    img_sz = in_bgr.shape[:2:]
    if img_sz[0] > tgt_size[0] or img_sz[1] > tgt_size[1]:
        # Making the ratio of the largest dimension with the given target size so there is no distortions
        idx = np.argmax(img_sz)
        ratio = tgt_size[idx] / img_sz[idx]
        out_bgr = cv2.resize(in_bgr, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    else:
        # If the image is smaller than the target size, we just return it
        out_bgr = in_bgr.copy()

    return out_bgr
