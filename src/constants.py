import os
import pathlib

IS_KAGGLE = os.getenv("KAGGLE") is not None
COMPE_NAME = ""

ROOT = pathlib.Path("/kaggle") if IS_KAGGLE else pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DATA_DIR = INPUT_DIR / "atmaCup#18_dataset"

TARGET_COLS = [
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "x_2",
    "y_2",
    "z_2",
    "x_3",
    "y_3",
    "z_3",
    "x_4",
    "y_4",
    "z_4",
    "x_5",
    "y_5",
    "z_5",
]
