"""
facerecog/trainer.py
Melatih LBPH Face Recognizer dari seluruh dataset yang tersedia.
"""
import os
import numpy as np
import cv2

from .config import DATASET_DIR, MODEL_PATH
from . import labels as lbl


def train() -> dict:
    """
    Latih model LBPH dari seluruh dataset.

    Returns:
        dict berisi informasi hasil training:
        {
            "total_images": int,
            "total_persons": int,
            "model_path": str
        }

    Raises:
        RuntimeError: Jika belum ada data terdaftar atau tidak ada gambar.
    """
    labels = lbl.load()
    if not labels:
        raise RuntimeError("Belum ada data terdaftar. Daftarkan wajah terlebih dahulu.")

    faces, ids = [], []
    for lid, name in labels.items():
        person_dir = os.path.join(DATASET_DIR, lid)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                ids.append(int(lid))

    if not faces:
        raise RuntimeError("Tidak ada gambar ditemukan di folder dataset.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write(MODEL_PATH)

    return {
        "total_images": len(faces),
        "total_persons": len(labels),
        "model_path": MODEL_PATH,
    }
