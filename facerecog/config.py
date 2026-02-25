"""
facerecog/config.py
All paths and constants used throughout the module.

Paths are resolved relative to the caller's working directory (os.getcwd()),
so dataset/, trainer/, and labels.json are always created inside the user's
project folder — not inside the facerecog package itself.
"""
import os
import cv2

# ─── Base Paths ───────────────────────────────────────────────────────────────
# Use the caller's working directory so the module is portable.
BASE_DIR    = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
MODEL_PATH  = os.path.join(TRAINER_DIR, "trainer.yml")

# ─── OpenCV Paths ─────────────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_PHOTOS           = 40    # photos per camera registration session
CONFIDENCE_THRESHOLD = 75    # LBPH confidence < this value = recognized
IMG_EXTS             = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Ensure required directories exist ───────────────────────────────────────
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
