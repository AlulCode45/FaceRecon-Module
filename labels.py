"""
facerecog/labels.py
Helper CRUD untuk file labels.json.
"""
import json
import os
from .config import LABELS_FILE


def load() -> dict:
    """Muat labels dari file JSON. Return dict kosong jika belum ada."""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return {}


def save(labels: dict) -> None:
    """Simpan labels ke file JSON."""
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def next_id(labels: dict) -> int:
    """Kembalikan ID berikutnya (max ID + 1)."""
    return max((int(k) for k in labels), default=0) + 1


def find_by_name(labels: dict, name: str) -> tuple[str | None, int | None]:
    """
    Cari label berdasarkan nama (case-insensitive).
    Return (lid_str, user_id_int) atau (None, None) jika tidak ditemukan.
    """
    for lid, lname in labels.items():
        if lname.lower() == name.lower():
            return lid, int(lid)
    return None, None
