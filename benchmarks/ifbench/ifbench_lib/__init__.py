"""IFBench helper library package."""

import os
from pathlib import Path

import nltk


NLTK_DATA_DIR = Path(__file__).parent / ".nltk_data"
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))
