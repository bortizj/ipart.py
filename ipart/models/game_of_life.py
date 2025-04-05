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

from tqdm import tqdm

from ipart.utils.imgproc import check_and_adjust_image_size

from ipart.utils.tools import GIFVideoMaker

# Global constants for the GoL algorithm
from ipart import TGT_SIZE

DEAD_TH: float = 1e-3


class GameOfLife:
    """
    Class to implement the Game of Life algorithm.
    The original algorithm for binary images:
        Birth:
            if unpopulated pixel surrounded by exactly n populated pixels, becomes populated next generation
        Death by loneliness:
            if populated pixel is surrounded by n - k or fewer, the pixel becomes unpopulated next generation
        Death by overpopulation:
            if populated pixel is surrounded by n + k or more, the pixel becomes unpopulated next generation

    Modification for color images:
        The difference between the pixel and the neighborhood is calculated using deltaE formula
        The neighborhood is characterized by the average and standard deviation of the pixels around the given pixel.

        Underpopulated neighborhood:
            if the standard deviation of the neighborhood is less than sigma[0]
        Overpopulation neighborhood:
            if the standard deviation of the neighborhood is greater than sigma[1]
        Death:
            The pixel will die if the distance between the pixel and the neighborhood is greater than mu.
        Death:
            The pixel will die if the neighborhood is overpopulated.
        Birth:
            The pixel will be born if the distance between the pixel and the neighborhood is less than mu or
            is already dead and the neighborhood is underpopulated.

        Death: set to a random color + noise
        Birth: set to the average of the neighborhood + noise
    """

    def __init__(
        self,
        in_bgr: Path,
        wsize: int = 5,
        mu: float = 3,
        sigma: tuple[float, float] = (5, 15),
        add_noise: tuple[float, float] = (10 / 255, 0.02),
        rng_seed: int = 42,
    ):
        # Create a random number generator with a seed, adds "predictable" uncertainty to the algorithm
        self.rng = np.random.default_rng(seed=rng_seed)

        # Getting the settings of the Game of Life algorithm: window size, mu, sigma, and dead threshold
        self.wsize = wsize
        self.mu = mu
        self.sigma = sigma

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
        self.segment_image()
        self.add_randomness(add_noise)

        # Converting to CIELab color space for better color distance calculations
        self.img_now = cv2.cvtColor(self.img_now, cv2.COLOR_BGR2Lab)

        # Creating a reference image for the generational error
        self.img_ref = self.img_now.copy()

    def play(self, path_gif, num_generations: int = 500, display: bool = True, play_fps: int = 60):
        """
        Plays the Game of Life algorithm for a given number of generations.
        """
        if path_gif is not None:
            gif = GIFVideoMaker(str(path_gif))
        tqdm_loop = tqdm(range(num_generations), desc="Game of Life", ncols=100)

        gen_error = []
        for _ in tqdm_loop:
            self.next_generation()

            # Getting the generational error
            gen_error.append(self.measure_generational_error())

            # Temporal color BGR image for displaying
            temp = cv2.cvtColor(self.img_now, cv2.COLOR_Lab2BGR)

            # Appends the current generation image to the gif
            if path_gif is not None:
                gif.append_frame(temp)

            # Displaying the current generation
            if display:
                cv2.imshow("Game of Life", temp)
                cv2.waitKey(int(1000 / play_fps))

            # Detecting if the population is stable
            diffs = np.diff(gen_error)
            if np.any(np.isclose(diffs, 0)):
                print(f"\nNo change detected: {diffs[-1]:.3f}")
                break

            # Getting the ratio of the differences between generations
            diff_ratios = np.abs(diffs[1:] / diffs[:-1])
            # A population is stable normally when the change ratio is higher than a threshold
            if np.any(diff_ratios > 7.5):
                print(f"\nPopulation is stable: {diff_ratios[-1]:.3f}")
                break

            if diff_ratios.size > 0:
                tqdm_loop.set_postfix(diff_ratio=diff_ratios[-1])

        # Destroying the display window
        if display:
            cv2.destroyAllWindows()

        # Storing all the generations in a gif
        if path_gif is not None:
            gif.make_gif_video()

    def measure_generational_error(self):
        """
        Measures the generational error of the algorithm by comparing the current generation with the reference image.
        """
        # Computing deltaE between the current generation and the reference image
        deltaE = cv2.sqrt(np.sum(cv2.pow(self.img_now - self.img_ref, 2), axis=2))

        return np.mean(deltaE)

    def segment_image(self):
        """
        Segments the image using kmeans.
        """
        smoothed_image = cv2.GaussianBlur(self.img_now, [13, 13], 1.5)
        pixels = smoothed_image.reshape((-1, 3)).astype(np.float32)

        # Define criteria for kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Creating an image where each cluster is represented by a random color (dead state)
        _, labels, centers = cv2.kmeans(pixels, 24, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        noise_a = self.rng.normal(0, 1, centers.shape).astype("float32") * np.array([[70, 127, 127]]).astype("float32")
        self.img_rand = noise_a[labels.flatten()].reshape(self.img_now.shape)

    def add_randomness(self, add_noise: tuple[float, float]):
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

        # Clipping the image between 0 and 1
        self.img_now = np.clip(self.img_now, 0, 1)

    def next_generation(self):
        """
        Computes the next generation of the GoL
        """
        # Initializing the next generation image
        img_next = self.img_now.copy()

        # Getting pixels that are dead in the current generation
        currently_dead = np.sum(np.abs(self.img_now), axis=2) <= DEAD_TH

        # Filtering the image to get the average of the neighborhood per channel
        ex1_nei = cv2.filter2D(self.img_now, -1, self.kernel, borderType=cv2.BORDER_REFLECT101)
        ex2_nei = cv2.filter2D(cv2.pow(self.img_now, 2), -1, self.kernel, borderType=cv2.BORDER_REFLECT101)

        # Verifying if the pixel will be alive or not:
        # the pixel is alive if is similar to the neighborhood |pixel - avg_n| <= mu
        # here the distance will be euclidean in the given color space
        deltaE_nei = cv2.sqrt(np.sum(cv2.pow(self.img_now - ex1_nei, 2), axis=2))
        will_die_due_difference = deltaE_nei > self.mu

        # Verifying if the neighborhood is over or under populated:
        # the neighborhood is overpopulated if the standard deviation of the neighborhood is greater than sigma[1]
        stdev = np.mean(cv2.sqrt(ex2_nei - cv2.pow(ex1_nei, 2)), axis=2)
        underpopulated = stdev < self.sigma[0]
        will_die_due_overpopulation = stdev > self.sigma[1]

        # Pixels that will given birth in the next generation or will die
        dead_will_die_diff = np.logical_or(currently_dead, will_die_due_difference)
        birth_pixels = np.logical_and(dead_will_die_diff, underpopulated)

        # The randomness of life is added to the pixels that will be born or dead in the next generation
        noise_a = self.rng.normal(0, self.sigma[1], self.img_now.shape).astype("float32")

        # Removing the overpopulated pixels from the current generation by setting them to a random color
        img_next[will_die_due_overpopulation, ::] = (
            self.img_rand[will_die_due_overpopulation, ::] + noise_a[will_die_due_overpopulation, ::]
        )

        # Adding the pixels that will be born in the next generation by adding noise to the neighborhood
        img_next[birth_pixels, ::] = ex1_nei[birth_pixels, ::] + noise_a[birth_pixels, ::] / (self.wsize**2 - 1)

        # Updating the current generation image
        self.img_now = img_next.copy()
