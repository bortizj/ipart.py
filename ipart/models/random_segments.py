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

from ipart.utils.imgproc import check_and_adjust_image_size

from ipart.utils.tools import GIFVideoMaker
from ipart.palettes.color_palettes import ColorPalette

from ipart import TGT_SIZE


class RandomSegments:
    """
    Class to implement random color segments algorithm for art generation.
    """

    def __init__(
        self,
        in_bgr: str,
        wsize: int = 7,
        num_smooth: int = 5,
        rng_seed: int = 42,
        color_palette: tuple[str, int] = ("kaggle", 24),
    ):
        # Create a random number generator with a seed, adds "predictable" uncertainty to the algorithm
        self.rng = np.random.default_rng(seed=rng_seed)

        # Settings of the algorithm
        self.wsize = wsize
        self.color_palette = color_palette

        # Creating the necessary kernel for the calculations
        self.kernel = np.ones((self.wsize, self.wsize)).astype("float32")
        self.kernel[int(self.wsize / 2), int(self.wsize / 2)] = 0
        self.kernel = self.kernel / (self.wsize**2 - 1)

        # Reading the image from the given path
        self.in_bgr = cv2.imread(str(in_bgr))

        # Resizing the image for computational efficiency
        self.img_now = check_and_adjust_image_size(self.in_bgr, tgt_size=TGT_SIZE)

        # Normalizing the image between 0 and 1
        self.img_now = self.img_now.astype("float32") / 255.0

        # Converting the image to Lab color space
        self.img_now = cv2.cvtColor(self.img_now, cv2.COLOR_BGR2Lab)

        # Filtering the image into homogeneous color regions (preserving color)
        self.segment_image(num_smooth)

    def segment_image(self, num_smooth: int):
        # Smoothing the image for a number of iterations
        for ii in range(num_smooth):
            # Filtering the image to get the average of the neighborhood per channel
            self.img_now = cv2.filter2D(self.img_now, -1, self.kernel, borderType=cv2.BORDER_REFLECT101)

        # Define criteria for kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Creating an image where each cluster is represented by a random color (dead state)
        pixels = self.img_now.reshape((-1, 3)).astype(np.float32)
        _, self.labels, self.colors = cv2.kmeans(pixels, self.color_palette[1], None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    def play(self, path_gif, display: bool = True, len_sec: float = 5.0, play_fps: int = 5, gif_fps: int = 10):
        """
        Plays the random segments algorithm.
        """
        if path_gif is not None:
            gif = GIFVideoMaker(str(path_gif), duration=int(1000 / gif_fps))
        else:
            gif = None

        self.img_now = self.colors[self.labels.astype("int")].reshape(self.img_now.shape)
        temp_ = cv2.cvtColor(self.img_now, cv2.COLOR_Lab2BGR)
        if display:
            cv2.imshow(f"Random segments", temp_)
            cv2.waitKey(int(1000 / play_fps))

        # Appends the current generation image to the gif
        if gif is not None:
            gif.append_frame(temp_)

        for __ in range(int(len_sec * play_fps)):
            # Selecting a new color palette
            cp = ColorPalette(self.rng, n_colors=self.color_palette[1], color_palette=self.color_palette[0])
            self.img_now_rand = cv2.cvtColor(cp.lut(self.labels).reshape(self.img_now.shape), cv2.COLOR_BGR2Lab)
            temp = cv2.cvtColor(self.img_now_rand, cv2.COLOR_Lab2BGR)
            if display:
                cv2.imshow(f"Random segments", temp)
                cv2.waitKey(int(1000 / play_fps))

            # Appends the current generation image to the gif
            if gif is not None:
                gif.append_frame(temp)

        if gif is not None:
            gif.append_frame(temp_)

        # Destroying the display window
        if display:
            cv2.destroyAllWindows()

        # Storing all the generations in a gif
        if path_gif is not None:
            gif.make_gif_video()
