"""
facerecog/users.py
Manajemen pengguna — list dan hapus.
"""
import os
import shutil
from . import labels as lbl
from .config import DATASET_DIR


def list_users() -> list[dict]:
    """
    Dapatkan daftar semua pengguna beserta jumlah foto dataset-nya.

    Returns:
        List of dict: [{"id": int, "name": str, "photos": int}, ...]
    """
    labels = lbl.load()
    result = []
    for lid, name in sorted(labels.items(), key=lambda x: int(x[0])):
        person_dir  = os.path.join(DATASET_DIR, lid)
        photo_count = 0
        if os.path.isdir(person_dir):
            photo_count = sum(1 for f in os.listdir(person_dir) if f.endswith(".jpg"))
        result.append({"id": int(lid), "name": name, "photos": photo_count})
    return result


def delete_user(name: str) -> dict:
    """
    Hapus pengguna beserta seluruh data fotonya.

    Args:
        name: Nama pengguna yang akan dihapus (case-insensitive).

    Returns:
        dict: {"id": int, "name": str, "photos_deleted": int}

    Raises:
        ValueError: Jika nama tidak ditemukan.
    """
    labels = lbl.load()
    lid_str, user_id = lbl.find_by_name(labels, name)

    if user_id is None:
        raise ValueError(f"Pengguna '{name}' tidak ditemukan.")

    person_dir    = os.path.join(DATASET_DIR, lid_str)
    photos_deleted = 0

    if os.path.isdir(person_dir):
        photos_deleted = sum(1 for f in os.listdir(person_dir) if f.endswith(".jpg"))
        shutil.rmtree(person_dir)

    del labels[lid_str]
    lbl.save(labels)

    return {"id": user_id, "name": name, "photos_deleted": photos_deleted}
