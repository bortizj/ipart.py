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

import os
from pathlib import Path

PKG_ROOT = Path(__file__).parent
REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DEBUG = os.environ.get("IPART_DEBUG") is not None

TGT_SIZE: tuple[int, int] = (512, 512)

__author__ = """Benhur Ortiz-Jaramillo"""
__email__ = "dukejob@gmail.com"
__version__ = "0.0.0"
__license__ = ""
