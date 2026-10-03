"""Shared paths, image size and file fingerprints."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
DATA = PROJECT / "data"
RUNS = PROJECT / "runs"
SIZE = (384, 512)  # Pillow uses (width, height).


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
