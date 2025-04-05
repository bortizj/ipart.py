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

from PIL import Image
import cv2

import numpy as np


class GIFVideoMaker:
    """
    Class to make a gif from images
    """

    def __init__(self, gif_path: str, duration: int = 100):
        """
        Initialize the GIFVideoMaker class.
        Parameters
        ----------
        gif_path: str
            Path to save the gif video.
        duration: int
            Duration of each frame in milliseconds.
        """
        self.gif_path = gif_path
        self.duration = duration
        self.frames = []

    def append_frame(self, frame: str):
        """
        Append a frame to the gif video.
        """
        if np.max(frame) > 1.0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = 255 * cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frames.append(Image.fromarray(frame.astype("uint8")))

    def make_gif_video(self):
        """
        Make a gif video from images.
        """
        self.frames[0].save(self.gif_path, save_all=True, append_images=self.frames[1:], duration=self.duration, loop=0)
