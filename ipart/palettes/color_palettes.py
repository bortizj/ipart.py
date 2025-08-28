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

from ipart import REPO_ROOT

DMC_FILE = REPO_ROOT.joinpath(r"ipart", r"assets", r"dmc2rgb.csv")
KAGGLE_FILE = REPO_ROOT.joinpath(r"ipart", r"assets", r"colors_kaggle.csv")
NEON_FILE = REPO_ROOT.joinpath(r"ipart", r"assets", r"neon.csv")


class ColorPalette:
    """
    Initializes the color palette.
    """

    def __init__(
        self, rng: np.random.Generator | None, color_palette: str | np.ndarray = "kaggle", n_colors: int | None = 100
    ):
        # Getting the list of colors from file or random generator
        if isinstance(color_palette, str):
            if color_palette == "random":
                if n_colors is None:
                    n_colors = 100
                rgb_code = rng.integers(0, 255, (n_colors, 3))
            else:
                if color_palette == "dmc":
                    __, rgb_code, __ = _get_dmc_colors()
                elif color_palette == "kaggle":
                    rgb_code, __ = _get_kaggle_colors()
                elif color_palette == "neon":
                    rgb_code = _get_neon_colors()
                else:
                    raise ValueError(f"Unknown color palette: {color_palette}")
                # Selecting the colors from the file
                if n_colors is None:
                    n_colors = rgb_code.shape[0]
                if n_colors > rgb_code.shape[0]:
                    n_colors = rgb_code.shape[0]

                # Even if selecting the whole vector we still want to shuffle it
                idx = rng.choice(rgb_code.shape[0], n_colors, replace=False)
                rgb_code = rgb_code[idx]

            # Normalizing and converting to opencv format
            self.n_colors = n_colors
            self.bgr_lut = rgb_code[::, ::-1].astype("float32") / 255.0
        elif isinstance(color_palette, np.ndarray):
            self.n_colors = color_palette.shape[0]
            self.bgr_lut = color_palette.copy()
        else:
            raise ValueError(f"Unknown color palette type: {type(color_palette)}")

    def lut(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the color palette to the image.
        Performs a look-up table transform of an array.
        """
        # Applying the color palette to the image
        return self.bgr_lut[img.astype("int")]


def _get_dmc_colors():
    df = pd.read_csv(DMC_FILE)
    data = df.values
    dmc_code = data[::, 0]
    rgb_code = data[::, 2:5].astype("float32")
    description = data[::, 1]

    return dmc_code, rgb_code, description


def _get_kaggle_colors():
    df = pd.read_csv(KAGGLE_FILE)
    data = df.values
    rgb_code = data[::, 3:].astype("float32")
    description = data[::, 1]

    return rgb_code, description


def _get_neon_colors():
    df = pd.read_csv(NEON_FILE)
    data = df.values
    rgb_code = data.astype("float32")

    return rgb_code
