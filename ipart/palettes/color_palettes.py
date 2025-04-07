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
import pandas as pd
import cv2

from ipart import REPO_ROOT

DMC_FILE = REPO_ROOT.joinpath(r"palettes", r"dmc2rgb.csv")
COLORS_FILE = REPO_ROOT.joinpath(r"palettes", r"colors_kaggle.csv")


class ColorPalette:
    """
    Initializes the color palette.
    """

    def __init__(self, rng: np.random.Generator, color_palette="kaggle", n_colors: int = 100):
        self.color_palette = color_palette

        # Getting the list of colors from file or random generator
        if color_palette == "random":
            rgb_code = rng.integers(0, 255, (n_colors, 3))
        else:
            if color_palette == "dmc":
                __, rgb_code, __ = get_dmc_colors()
            elif color_palette == "kaggle":
                rgb_code, __ = get_colors_list()
            else:
                raise ValueError(f"Unknown color palette: {color_palette}")
            # Selecting the colors from the file
            if n_colors > rgb_code.shape[0]:
                n_colors = rgb_code.shape[0]

            # Even if selecting the whole vector we still want to shuffle it
            idx = rng.choice(rgb_code.shape[0], n_colors, replace=False)
            rgb_code = rgb_code[idx]

        # Normalizing and converting to opencv format
        self.bgr_lut = rgb_code[::, ::-1].astype("float32") / 255.0

    def lut(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the color palette to the image.
        Performs a look-up table transform of an array.
        """
        # Applying the color palette to the image
        return self.bgr_lut[img.astype("int")]


def get_dmc_colors():
    df = pd.read_csv(DMC_FILE)
    data = df.values
    dmc_code = data[::, 0]
    rgb_code = data[::, 2:5].astype("float32")
    description = data[::, 1]

    return dmc_code, rgb_code, description


def get_colors_list():
    df = pd.read_csv(COLORS_FILE)
    data = df.values
    rgb_code = data[::, 3:].astype("float32")
    description = data[::, 1]

    return rgb_code, description
