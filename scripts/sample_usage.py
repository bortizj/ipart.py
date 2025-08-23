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

import time

import cv2
import numpy as np

from ipart import REPO_ROOT
from ipart.models.base_stroke import BaseStroke
from ipart.models.game_of_life import GameOfLife
from ipart.models.lbp import LBP
from ipart.models.random_segments import RandomSegments

PATH_SAMPLES = REPO_ROOT.joinpath(r"data")
GIF_FPS = 10
DISP_FPS = 10


def fun(blk):
    return np.ones_like(blk) * blk.mean(axis=(0, 1), keepdims=True)


if __name__ == "__main__":
    # Example test for the sample images
    for ii in range(5):
        path_img = PATH_SAMPLES.joinpath(f"test_{ii}.jpg")
        path_gol = PATH_SAMPLES.joinpath(f"test_{ii}_gol.gif")
        path_lbp = PATH_SAMPLES.joinpath(f"test_{ii}_lbp.gif")
        path_rs = PATH_SAMPLES.joinpath(f"test_{ii}_rs.gif")
        img = cv2.imread(str(path_img))

        # Random seed from the current time
        bs = BaseStroke(path_img, fun, rng_seed=int(time.time()) + 42)
        bs.play(None, play_fps=DISP_FPS, gif_fps=GIF_FPS)
        gol = GameOfLife(path_img, rng_seed=int(time.time()) + 42)
        lbp = LBP(path_img, rng_seed=int(time.time()) + 42)
        rs = RandomSegments(path_img, rng_seed=int(time.time()) + 42)
        lbp.play(None, play_fps=DISP_FPS, gif_fps=GIF_FPS)
        gol.play(None, play_fps=DISP_FPS, gif_fps=GIF_FPS)
        rs.play(None, play_fps=DISP_FPS, gif_fps=1)
