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


def make_gaussian_color_in_block(blk_labels: np.ndarray, cp: ColorPalette) -> np.ndarray:
    # Making a Gaussian kernel
    wsize = max(blk_labels.shape[0], blk_labels.shape[1])
    sigma = (wsize - 1) / 2.0
    gauss_kernel = cv2.getGaussianKernel(wsize, sigma=sigma)
    gauss_kernel = gauss_kernel * gauss_kernel.T
    gauss_kernel = gauss_kernel[:, :, np.newaxis] / gauss_kernel.max()
    gauss_kernel = gauss_kernel[: blk_labels.shape[0], : blk_labels.shape[1]]

    # Getting the mos common color in the block
    counts = np.bincount(blk_labels.flatten())
    color_id = int(np.argmax(counts))

    # Making the color a gradient using the Gaussian shape
    blk_circle = color_id * np.ones_like(blk_labels)
    blk_circle = cp.lut(blk_circle)
    blk_circle = (blk_circle * gauss_kernel).astype("float32")

    return blk_circle


def make_circle_in_block(blk_labels: np.ndarray, cp: ColorPalette) -> np.ndarray:
    # Getting the mos common color in the block
    counts = np.bincount(blk_labels.flatten())
    color_id = int(np.argmax(counts))

    # Making the color a gradient using the Gaussian shape
    cv2.circle(
        blk_labels,
        (int(blk_labels.shape[1] / 2), int(blk_labels.shape[0] / 2)),
        int(blk_labels.shape[0] / 2),
        color_id,
        -1,
    )
    blk_circle = cp.lut(blk_labels)

    return blk_circle


class BaseStroke:
    """
    Base class for stroke-based image processing.
    """

    def __init__(
        self,
        in_bgr: Path,
        func: Callable[[np.ndarray, ColorPalette], np.ndarray],
        wsize: int = 13,
        overlap_factor: float = 0.0,
        color_palette: tuple[str, int] = ("kaggle", 24),
        rng_seed: int = 42,
    ):
        # Create a random number generator with a seed, adds "predictable" uncertainty to the algorithm
        self.rng = np.random.default_rng(seed=rng_seed)

        # Settings of the algorithm
        self.wsize = wsize
        self.color_palette = color_palette
        self.overlap_factor = overlap_factor
        self.func = func

        # Creating the necessary kernel for the calculations
        self.kernel = np.ones((self.wsize, self.wsize)).astype("float32")
        self.kernel = self.kernel / (self.wsize**2)

        # Reading the image from the given path
        self.in_bgr = cv2.imread(str(in_bgr))

        if self.in_bgr is None:
            print("ERROR: Not possible to read image!")
            return

        # Resizing the image for computational efficiency
        self.img_now = check_and_adjust_image_size(self.in_bgr, tgt_size=TGT_SIZE)

        # Normalizing the image between 0 and 1
        self.img_now = self.img_now.astype("float32") / 255.0

        # Converting the image to Lab color space
        self.img_process = cv2.cvtColor(self.img_now, cv2.COLOR_BGR2Lab)

        # Filtering the image into homogeneous color regions (preserving color)
        self.segment_image()

        # Getting the color palette for the image
        if self.color_palette[0] == "same":
            self.cp = ColorPalette(self.rng, color_palette=self.colors)
        else:
            self.cp = ColorPalette(self.rng, n_colors=self.color_palette[1], color_palette=self.color_palette[0])

        # getting the image in the given color palette
        self.img_process = self.cp.lut(self.labels).reshape(self.img_process.shape)
        self.img_strokes = self.img_process.copy()
        self.labels = self.labels.reshape(self.img_process.shape[0:2])

    def compute_image_gradient(self):
        """
        Computes the gradient of the image.
        """
        # TODO: Here get the local gradient patches that will be applied to the strokes
        self.img_gradient = cv2.Sobel(self.img_process, cv2.CV_64F, 1, 1, ksize=5)
        self.img_gradient = cv2.convertScaleAbs(self.img_gradient)

    def compute_image_texture(self):
        """
        Computes the texture of the image.
        """
        # TODO: Here get the local texture patches that will be applied to the strokes
        self.img_texture = cv2.Laplacian(self.img_process, cv2.CV_64F)
        self.img_texture = cv2.convertScaleAbs(self.img_texture)

    def segment_image(self):
        """
        Segments the image into homogeneous color regions.
        """
        # Filtering the image to get the average of the neighborhood per channel
        self.img_process = cv2.filter2D(self.img_process, -1, self.kernel, borderType=cv2.BORDER_REFLECT101)

        # Define criteria for kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Creating an image where each cluster is represented by a random color
        pixels = self.img_process.reshape((-1, 3)).astype("float32")

        best_labels = np.zeros((pixels.shape[0], 1), dtype="int32")
        _, self.labels, self.colors = cv2.kmeans(
            pixels, self.color_palette[1], best_labels, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        self.colors = cv2.cvtColor(self.colors[np.newaxis, ::], cv2.COLOR_Lab2BGR)[0, ::]

    def play(self, path_gif, display: bool = True, play_fps: int = 60, gif_fps: int = 10):
        """
        Plays the stroke base class.
        """
        if path_gif is not None:
            gif = GIFVideoMaker(str(path_gif), duration=int(1000 / gif_fps))
        else:
            gif = None

        # Getting image and block sizes
        n_rows, n_cols, __ = self.img_process.shape
        hwsize = int(self.wsize / 2)

        # A value between 0 (no overlap) and 1 (100% overlap).
        step_size = int(self.wsize * (1 - self.overlap_factor))
        if step_size == 0:
            step_size = 1

        tqdm_loop_ii = tqdm(range(hwsize, n_rows - hwsize + step_size, step_size), desc="Basic stroke row", ncols=100)
        tqdm_loop_jj = tqdm(range(hwsize, n_cols - hwsize + step_size, step_size), desc="Basic stroke col", ncols=100)

        for ii in tqdm_loop_ii:
            for jj in tqdm_loop_jj:
                # Checking that we are not outside the image
                edge_ii = min(ii + hwsize + 1, n_rows)
                edge_jj = min(jj + hwsize + 1, n_cols)

                # Getting the information of the current block
                curr_img_strokes = self.img_strokes[ii - hwsize : edge_ii, jj - hwsize : edge_jj]
                curr_blk = np.zeros_like(curr_img_strokes)

                # Applying the function to the current block
                curr_blk_labels = self.labels[ii - hwsize : edge_ii, jj - hwsize : edge_jj]
                curr_blk = self.func(curr_blk_labels, self.cp)

                curr_img_strokes = cv2.addWeighted(
                    curr_img_strokes,
                    0.25,
                    curr_blk,
                    1.0,
                    0,
                )

                # Trying to clean the black border
                # curr_img_strokes[np.where(curr_blk_labels != -1)] = curr_img_strokes.mean(axis=(0, 1), keepdims=True)
                self.img_strokes[ii - hwsize : ii + hwsize + 1, jj - hwsize : jj + hwsize + 1] = curr_img_strokes.copy()
                self.img_now[ii - hwsize : ii + hwsize + 1, jj - hwsize : jj + hwsize + 1] = curr_img_strokes.copy()

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
        if gif is not None:
            gif.make_gif_video()
