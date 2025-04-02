import os
from pathlib import Path

PKG_ROOT = Path(__file__).parent
REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DEBUG = os.environ.get("IPART_DEBUG") is not None

__author__ = """Benhur Ortiz-Jaramillo"""
__email__ = "dukejob@gmail.com"
__version__ = "0.0.0"
__license__ = ""
