"""
facerecog/dataset.py
Registrasi wajah — dari kamera real-time atau dari file/folder gambar.
"""
import os
import shutil
import cv2

from .config import (
    DATASET_DIR, CASCADE_PATH, MAX_PHOTOS, IMG_EXTS
)
from . import labels as lbl


# ─── Internal Helper ──────────────────────────────────────────────────────────

def _prepare_user(name: str, overwrite: bool = False, append: bool = True) -> tuple[int, str, dict]:
    """
    Siapkan: cek nama, tentukan ID, buat folder dataset.

    Returns:
        (user_id, person_dir, labels_dict)

    Raises:
        ValueError: jika nama kosong atau overwrite/append ditolak.
    """
    if not name:
        raise ValueError("Nama tidak boleh kosong.")

    labels  = lbl.load()
    lid_str, user_id = lbl.find_by_name(labels, name)

    if user_id is not None:
        person_dir = os.path.join(DATASET_DIR, lid_str)
        if not append and not overwrite:
            raise ValueError(f"'{name}' sudah terdaftar (ID {user_id}). Set overwrite=True atau append=True.")
        if overwrite:
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
        # append: biarkan folder apa adanya
    else:
        user_id = lbl.next_id(labels)
        lid_str = str(user_id)
        labels[lid_str] = name

    person_dir = os.path.join(DATASET_DIR, str(user_id))
    os.makedirs(person_dir, exist_ok=True)
    return user_id, person_dir, labels


def _count_existing(person_dir: str) -> int:
    """Hitung jumlah file .jpg yang sudah ada di folder."""
    return sum(1 for f in os.listdir(person_dir) if f.endswith(".jpg"))


# ─── Public API ───────────────────────────────────────────────────────────────

def register_from_camera(
    name: str,
    overwrite: bool = False,
    append: bool = True,
    max_photos: int = MAX_PHOTOS,
    camera_index: int = 0,
    app_name: str = "Face Recognition",
) -> int:
    """
    Capture face photos from camera and save as training data.

    Args:
        name        : Person's name to register.
        overwrite   : Delete old dataset before saving.
        append      : Add to existing dataset.
        max_photos  : Number of photos to capture.
        camera_index: Camera index (default 0).
        app_name    : Application name shown in window title.

    Returns:
        Jumlah foto yang berhasil disimpan.

    Raises:
        ValueError  : Jika nama kosong atau konflik overwrite/append.
        RuntimeError: Jika kamera tidak bisa dibuka.
    """
    user_id, person_dir, labels = _prepare_user(name, overwrite=overwrite, append=append)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka kamera (index {camera_index}).")

    count   = _count_existing(person_dir)
    saved   = 0
    WIN     = f"Register Face — {app_name}"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while saved < max_photos:
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            if saved >= max_photos:
                break
            count += 1
            saved += 1
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(frame, f"{saved}/{max_photos}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

        cv2.putText(frame, f"Registering: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)
        cv2.putText(frame, f"Photos: {saved}/{max_photos}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
        cv2.putText(frame, "Press 'q' / Esc to cancel", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(80) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved > 0:
        lbl.save(labels)

    return saved


def register_from_image(
    name: str,
    src: str,
    overwrite: bool = False,
    append: bool = True,
) -> int:
    """
    Daftarkan wajah dari file gambar tunggal atau folder berisi banyak gambar.

    Args:
        name     : Nama orang yang didaftarkan.
        src      : Path ke file gambar atau folder.
        overwrite: Hapus dataset lama sebelum menyimpan.
        append   : Tambah ke dataset yang sudah ada.

    Returns:
        Jumlah foto wajah yang berhasil disimpan.

    Raises:
        ValueError  : Jika nama kosong, path tidak ada, atau tidak ada gambar.
        RuntimeError: Jika tidak ada wajah berhasil disimpan.
    """
    if not os.path.exists(src):
        raise ValueError(f"Path tidak ditemukan: {src}")

    user_id, person_dir, labels = _prepare_user(name, overwrite=overwrite, append=append)

    # Kumpulkan file gambar
    if os.path.isfile(src):
        img_files = [src] if os.path.splitext(src)[1].lower() in IMG_EXTS else []
    else:
        img_files = [
            os.path.join(src, f)
            for f in sorted(os.listdir(src))
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]

    if not img_files:
        raise ValueError("Tidak ada file gambar ditemukan di path yang diberikan.")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    count = _count_existing(person_dir)
    saved = 0
    skipped = 0

    for img_path in img_files:
        frame = cv2.imread(img_path)
        if frame is None:
            skipped += 1
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            skipped += 1
            continue

        for (x, y, w, h) in faces:
            count += 1
            saved += 1
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), gray[y:y + h, x:x + w])

    if saved == 0:
        # Bersihkan label baru jika tidak ada yang tersimpan
        if str(user_id) in labels and not os.listdir(person_dir):
            del labels[str(user_id)]
        lbl.save(labels)
        raise RuntimeError("Tidak ada wajah berhasil disimpan dari gambar yang diberikan.")

    lbl.save(labels)
    return saved
