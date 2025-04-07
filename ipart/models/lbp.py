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
from ipart import REPO_ROOT

LBP_FILE = REPO_ROOT.joinpath(r"ipart", r"assets", r"lbp")
RADII = {1: 8, 1.5: 12}
LINE_MAP = {8: 0, 12: 1}
BETA_BLEND = 0.75


class LBP:
    """
    Class to implement the LBP algorithm for texture analysis.
    Estimates the local binary patterns in a radius r around the central pixel based on
    Maenpaa, "The Local Binary Pattern Approach To Texture Analysis: Extensions And Applications." PhD thesis,
    University of Oulu. (2003)
    """

    def __init__(
        self,
        in_bgr: str,
        r: float = 1.5,
        th: float = 10 / 255.0,
        rng_seed: int = 42,
        color_palette: str = "neon",
    ):
        # Create a random number generator with a seed, adds "predictable" uncertainty to the algorithm
        self.rng = np.random.default_rng(seed=rng_seed)

        # The setting of the parameters for the LBP algorithm
        self.r = r
        self.th = th

        # Reading the image from the given path
        self.in_bgr = cv2.imread(str(in_bgr))

        # Resizing the image for computational efficiency
        self.img_now = check_and_adjust_image_size(self.in_bgr, tgt_size=TGT_SIZE)

        self.img_ref = self.img_now.copy()

        # Reading the LBP table
        # Look at tables from previous investigation see Maenpaa work
        self.n = RADII[self.r]
        with open(LBP_FILE, encoding="utf-8-sig") as f:
            lines = f.readlines()
            table = lines[LINE_MAP[self.n]].split(",")
            self.table = np.array(list(map(int, table))).astype("int")

        self.n_patterns = len(np.unique(self.table))

        self.color_palette = ColorPalette(self.rng, n_colors=self.n_patterns, color_palette=color_palette)

        # If the image is BGR then we convert it to grayscale
        if len(self.img_now.shape) > 2:
            self.img_now = cv2.cvtColor(self.img_now, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        else:
            self.img_now = self.img_now.astype("float32") / 255.0

    def play(self, path_gif, display: bool = True, play_fps: int = 5, gif_fps: int = 10) -> np.ndarray:

        def blend_img(table, lbp_img, img):
            temp = cv2.medianBlur(table[lbp_img.astype("int")].astype("uint8"), 5)
            bgr_lbp = (255 * self.color_palette.lut(temp)).astype("uint8")

            # Blending the images
            bgr_lbp = cv2.addWeighted(bgr_lbp, BETA_BLEND, img, 1.0, 0)
            return bgr_lbp

        if path_gif is not None:
            gif = GIFVideoMaker(str(path_gif), duration=int(1000 / gif_fps))

        lbp_img = np.zeros_like(self.img_now)

        # Storing the initial image for the gif
        if path_gif is not None:
            for jj in range(self.n_patterns):
                gif.append_frame(self.img_ref)

        # Pixels located outside the grid are computed using bilinear interpolation
        xx, yy = np.meshgrid(
            np.arange(0, self.img_now.shape[1]).astype("float32"),
            np.arange(0, self.img_now.shape[0]).astype("float32"),
        )
        for ii in range(self.n):
            xi = xx - self.r * np.sin(2 * np.pi * ii / self.n)
            yi = yy + self.r * np.cos(2 * np.pi * ii / self.n)

            # Comparison against neighbor
            img_ii = cv2.remap(self.img_now, xi, yi, cv2.INTER_LINEAR)
            __, img_ii = cv2.threshold(np.abs(self.img_now - img_ii), self.th, 1, cv2.THRESH_BINARY)

            # Accumulating pattern code
            lbp_img += img_ii.astype("float32") * (2**ii)

            # Blending the images
            bgr_lbp = blend_img(self.table, lbp_img, self.img_ref)

            # Appends the current texture pattern image to the gif
            if path_gif is not None:
                for jj in range(max(1, int(self.n_patterns / 3))):
                    gif.append_frame(bgr_lbp)

            if display:
                # Displaying the intermediate images
                cv2.imshow(f"LBP", bgr_lbp)
                cv2.waitKey(int(1000 / play_fps))

        # Blending the images
        bgr_lbp = blend_img(self.table, lbp_img, self.img_ref)

        # Storing the gif
        if path_gif is not None:
            for ii in range(self.n_patterns):
                gif.append_frame(bgr_lbp)
            gif.make_gif_video()

        if display:
            # Displaying the intermediate images
            cv2.imshow(f"LBP", bgr_lbp)
            cv2.waitKey(int(self.n_patterns * 1000 / play_fps))
            cv2.destroyAllWindows()
