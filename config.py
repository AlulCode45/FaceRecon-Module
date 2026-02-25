"""
facerecog/config.py
Semua path dan konstanta yang digunakan di seluruh modul.
"""
import os
import cv2

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
MODEL_PATH  = os.path.join(TRAINER_DIR, "trainer.yml")

# ─── OpenCV Paths ─────────────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ─── Konstanta ────────────────────────────────────────────────────────────────
MAX_PHOTOS      = 40           # jumlah foto per sesi register kamera
CONFIDENCE_THRESHOLD = 75      # LBPH confidence < nilai ini = dikenali
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Pastikan folder wajib ada ────────────────────────────────────────────────
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
