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

from ipart.models.game_of_life import GameOfLife
from ipart.models.lbp import LBP
from ipart import REPO_ROOT
import time

import cv2


PATH_SAMPLES = REPO_ROOT.joinpath(r"data")

if __name__ == "__main__":
    # Example test for the sample images
    for ii in range(5):
        path_img = PATH_SAMPLES.joinpath(f"test_{ii}.jpg")
        path_gol = PATH_SAMPLES.joinpath(f"test_{ii}_gol.gif")
        path_lbp = PATH_SAMPLES.joinpath(f"test_{ii}_lbp.gif")
        img = cv2.imread(str(path_img))

        # Random seed from the current time
        gol = GameOfLife(path_img, rng_seed=int(time.time()) + 42)
        lbp = LBP(path_img, rng_seed=int(time.time()) + 42)
        lbp.play(None, play_fps=10)
        gol.play(None, play_fps=10)
