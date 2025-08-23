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

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

# Global constants for the GoL algorithm
from ipart import TGT_SIZE
from ipart.palettes.color_palettes import ColorPalette
from ipart.utils.imgproc import check_and_adjust_image_size
from ipart.utils.tools import GIFVideoMaker


class BaseStroke:
    """
    Base class for stroke-based image processing.
    """

    def __init__(
        self,
        in_bgr: Path,
        func: Callable[[np.ndarray], np.ndarray],
        wsize: int = 21,
        overlap_factor: float = 0.5,
        color_palette: str = "kaggle",
        rng_seed: int = 42,
    ):
        # Create a random number generator with a seed, adds "predictable" uncertainty to the algorithm
        self.rng = np.random.default_rng(seed=rng_seed)

        # Reading the image from the given path
        self.in_bgr = cv2.imread(str(in_bgr))

        # Getting the settings of the algorithm
        self.wsize = wsize
        self.overlap_factor = overlap_factor
        self.func = func

        # Resizing the image for computational efficiency
        self.img_now = check_and_adjust_image_size(self.in_bgr, tgt_size=TGT_SIZE)

        # Normalizing the image between 0 and 1
        self.img_now = self.img_now.astype("float32") / 255.0

        # Creating the color palette for the algorithm
        self.color_palette = ColorPalette(self.rng, n_colors=None, color_palette=color_palette)
        self.n_colors = self.color_palette.n_colors

    def play(self, path_gif, display: bool = True, play_fps: int = 60, gif_fps: int = 10):
        """
        Plays the stroke base class.
        """
        if path_gif is not None:
            gif = GIFVideoMaker(str(path_gif), duration=int(1000 / gif_fps))
        else:
            gif = None

        # Making a copy of the current image
        img_copy = self.img_now.copy()

        # Getting image and block sizes
        n_rows, n_cols, __ = self.img_now.shape
        hwsize = int(self.wsize / 2)

        # A value between 0 (no overlap) and 1 (100% overlap).
        step_size = int(self.wsize * (1 - self.overlap_factor))
        if step_size == 0:
            step_size = 1

        tqdm_loop_ii = tqdm(range(hwsize, n_rows - hwsize + 1, step_size), desc="Basic stroke row", ncols=100)
        tqdm_loop_jj = tqdm(range(hwsize, n_cols - hwsize + 1, step_size), desc="Basic stroke col", ncols=100)

        for ii in tqdm_loop_ii:
            for jj in tqdm_loop_jj:
                # Applying the function to the current block
                curr_blk = img_copy[ii - hwsize : ii + hwsize + 1, jj - hwsize : jj + hwsize + 1]
                curr_blk = self.func(curr_blk)
                self.img_now[ii - hwsize : ii + hwsize + 1, jj - hwsize : jj + hwsize + 1] = curr_blk

                # Appends the current generation image to the gif
                if gif is not None:
                    gif.append_frame(self.img_now)

                # Displaying the current generation
                if display:
                    cv2.imshow("Strokes", self.img_now)
                    cv2.waitKey(int(1000 / play_fps))

        # Destroying the display window
        if display:
            cv2.destroyAllWindows()

        # Storing all the generations in a gif
        if path_gif is not None:
            gif.make_gif_video()
