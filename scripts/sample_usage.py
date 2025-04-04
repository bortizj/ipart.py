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
from ipart.utils.tools import GIFVideoMaker
from ipart import REPO_ROOT

import cv2

PATH_SAMPLES = REPO_ROOT.joinpath(r"data")

if __name__ == "__main__":
    # Example test for the sample images
    for ii in range(5):
        path_img = PATH_SAMPLES.joinpath(f"test_{ii}.jpg")
        path_vid = PATH_SAMPLES.joinpath(f"test_{ii}_gol.gif")
        img = cv2.imread(str(path_img))

        gol = GameOfLife(path_img)
        gif = GIFVideoMaker(str(path_vid))
        for jj in range(100):
            gol.next_generation()

            temp = cv2.cvtColor(gol.img_now, cv2.COLOR_Lab2BGR)

            gif.append_frame(temp)

            cv2.imshow("Game of Life", temp)
            cv2.waitKey(100)

        gif.make_gif_video()
