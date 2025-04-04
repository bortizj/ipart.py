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
from pathlib import Path

from ipart.utils.imgproc import check_and_adjust_image_size


class GameOfLife:
    """
    Class to implement the Game of Life algorithm.
    The algorithm
        Birth:
            if unpopulated pixel surrounded by exactly n populated pixels, becomes populated next generation
        Death by loneliness:
            if populated pixel is surrounded by n - k or fewer, the pixel becomes unpopulated next generation
        Death by overpopulation:
            if populated pixel is surrounded by n + k or more, the pixel becomes unpopulated next generation
    Modification for color images:
        The distance between the pixel and the neighborhood is calculated using the euclidean distance in the given color space.
        The neighborhood is defined as the average of the pixels in the neighborhood.
        The standard deviation of the neighborhood is calculated using the average of the pixels in the neighborhood.
        Underpopulated:
            if the standard deviation of the neighborhood is less than sigma[0]
        Overpopulation:
            if the standard deviation of the neighborhood is greater than sigma[1], the pixel will die
        Death:
            The pixel will die if the distance between the pixel and the neighborhood is greater than mu.
        Death:
            The pixel will die if the neighborhood is overpopulated.
        Birth:
            The pixel will be born if the distance between the pixel and the neighborhood is less than mu or is dead
            and the neighborhood is underpopulated.

        Death: set to a random color
        Birth: set to the average of the neighborhood + noise
    """

    def __init__(
        self,
        in_bgr: Path,
        wsize: int = 3,
        dead_th: float = 1e-3,
        mu: float = 10,
        sigma: tuple[float, float] = (5, 50),
        tgt_size: tuple[int, int] = (360, 360),
        add_noise: tuple[float, float] = (5 / 255, 0.01),
        color_space: str = "Lab",
        rng_seed: int = 42,
    ):
        # Create a random number generator with a seed, adds "predictable" uncertainty to the algorithm
        self.rng = np.random.default_rng(seed=rng_seed)

        # Getting the settings of the Game of Life algorithm
        self.wsize = wsize

        # Setting the threshold for the alive pixels
        self.mu = mu
        self.sigma = sigma
        self.dead_th = dead_th

        # Crating necessary kernel for the calculations
        self.kernel = np.ones((self.wsize, self.wsize)).astype("float32")
        self.kernel[int(self.wsize / 2), int(self.wsize / 2)] = 0
        self.kernel = self.kernel / (self.wsize**2 - 1)

        # Reading the image from the given path
        self.in_bgr = cv2.imread(str(in_bgr))

        # Adjusting the image size to be smaller for computational purposes
        self.img_now = check_and_adjust_image_size(self.in_bgr, tgt_size=tgt_size)

        # Normalizing the image to be between 0 and 1
        self.img_now = self.img_now.astype("float32") / 255.0
        self.add_randomness(add_noise)

        # Converting to CIELAB color space for better color distance calculations
        if color_space == "Lab":
            self.img_now = cv2.cvtColor(self.img_now, cv2.COLOR_BGR2Lab)

    def segment_image(self):
        """
        Segments the image using kmeans clustering.
        """
        pass

    def add_randomness(self, add_noise: list[float, float]):
        """
        Adds randomness to the image by adding gaussian noise and salt and pepper noise.
        """
        # Adding gaussian noise to the image
        self.img_now = self.img_now + self.rng.normal(0, add_noise[0], self.img_now.shape).astype("float32")

        # Adding salt and pepper noise
        num_noisy_px = int(add_noise[1] * self.img_now.size / 2)
        # Adding pepper noise
        ii = self.rng.integers(0, self.img_now.shape[0] - 1, num_noisy_px)
        jj = self.rng.integers(0, self.img_now.shape[1] - 1, num_noisy_px)
        self.img_now[ii, jj, :] = 0
        # Adding salt noise
        ii = self.rng.integers(0, self.img_now.shape[0] - 1, num_noisy_px)
        jj = self.rng.integers(0, self.img_now.shape[1] - 1, num_noisy_px)
        self.img_now[ii, jj, :] = 1

        # Clipping the image to be between 0 and 1
        self.img_now = np.clip(self.img_now, 0, 1)

    def next_generation(self):
        """
        Computes the next generation of the GoL
        """
        # Initializing the next generation image
        img_next = self.img_now.copy()

        # Getting pixels that are dead in the current generation
        currently_dead = np.sum(np.abs(self.img_now), axis=2) <= self.dead_th

        # Filtering the image to get the average of the neighborhood per channel
        ex1_nei = cv2.filter2D(self.img_now, -1, self.kernel, borderType=cv2.BORDER_REFLECT101)
        ex2_nei = cv2.filter2D(cv2.pow(self.img_now, 2), -1, self.kernel, borderType=cv2.BORDER_REFLECT101)

        # Verifying if the pixel will be alive or not:
        # the pixel is alive if is similar to the neighborhood |pixel - avg_n| <= mu
        # here the distance will be euclidean in the given color space
        euc_nei = cv2.sqrt(np.sum(cv2.pow(self.img_now - ex1_nei, 2), axis=2))
        will_die_due_difference = euc_nei > self.mu

        # Verifying if the neighborhood is over or under populated:
        # the neighborhood is overpopulated if the standard deviation of the neighborhood is greater than sigma[1]
        stdev = np.sum(cv2.sqrt(ex2_nei - cv2.pow(ex1_nei, 2)), axis=2)
        underpopulated = stdev < self.sigma[0]
        will_die_due_overpopulation = stdev > self.sigma[1]

        # Pixels that will given birth in the next generation or will die
        dead_will_die_diff = np.logical_or(currently_dead, will_die_due_difference)
        birth_pixels = np.logical_and(dead_will_die_diff, underpopulated)

        # Removing the overpopulated pixels from the current generation by setting them to a random color
        noise_a = self.rng.normal(0, 1, self.img_now.shape).astype("float32")
        noise_a = noise_a * np.array([100, 128, 128]).reshape((1, 1, 3)).astype("float32")
        img_next[will_die_due_overpopulation, ::] = noise_a[will_die_due_overpopulation, ::]

        # Adding the pixels that will be born in the next generation by adding noise to the neighborhood
        noise_a = self.rng.normal(0, self.sigma[1], self.img_now.shape).astype("float32")
        img_next[birth_pixels, ::] = ex1_nei[birth_pixels, ::] + noise_a[birth_pixels, ::]

        # Updating the current generation image
        self.img_now = img_next.copy()

        temp = cv2.cvtColor(img_next, cv2.COLOR_Lab2BGR)

        cv2.imshow("Game of Life", temp)
        cv2.waitKey(250)
