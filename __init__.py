"""
facerecog/__init__.py
High-level API — semua fitur cukup lewat class FaceRecog.

Contoh penggunaan cepat:

    from facerecog import FaceRecog

    fr = FaceRecog()

    # Daftarkan via kamera
    fr.register_from_camera("Budi")

    # Daftarkan via gambar/folder
    fr.register_from_image("Budi", "/path/ke/foto_budi/")

    # Latih model
    fr.train()

    # Deteksi dari kamera (real-time)
    fr.detect_camera()

    # Deteksi dari gambar
    result = fr.detect_image("/path/foto.jpg", show=True)
    for face in result.faces:
        print(face.name, face.score)

    # Daftar pengguna
    users = fr.list_users()

    # Hapus pengguna
    fr.delete_user("Budi")
"""

from .config import (
    BASE_DIR, DATASET_DIR, TRAINER_DIR,
    LABELS_FILE, MODEL_PATH, CASCADE_PATH,
    MAX_PHOTOS, CONFIDENCE_THRESHOLD,
)
from . import labels  as _labels_mod
from . import dataset as _dataset_mod
from . import trainer as _trainer_mod
from . import detector as _detector_mod
from . import users   as _users_mod

from .detector import DetectionResult, FaceResult


class FaceRecog:
    """
    Antarmuka utama modul pengenalan wajah ecolube.id.

    Semua operasi (register, train, detect, manage user) tersedia
    sebagai method di class ini.
    """

    # ── Konfigurasi ──────────────────────────────────────────────────────────

    def __init__(
        self,
        threshold: int = CONFIDENCE_THRESHOLD,
        max_photos: int = MAX_PHOTOS,
        camera_index: int = 0,
    ):
        """
        Args:
            threshold   : Batas confidence LBPH (default 75). Lebih rendah = lebih ketat.
            max_photos  : Jumlah foto per sesi registrasi kamera (default 40).
            camera_index: Index kamera yang digunakan (default 0).
        """
        self.threshold    = threshold
        self.max_photos   = max_photos
        self.camera_index = camera_index

    # ── Registrasi ───────────────────────────────────────────────────────────

    def register_from_camera(
        self,
        name: str,
        overwrite: bool = False,
        append: bool = True,
    ) -> int:
        """
        Daftarkan wajah baru via kamera.

        Args:
            name     : Nama orang.
            overwrite: Hapus data lama sebelum menyimpan.
            append   : Tambah ke data yang sudah ada.

        Returns:
            Jumlah foto yang tersimpan.
        """
        saved = _dataset_mod.register_from_camera(
            name=name,
            overwrite=overwrite,
            append=append,
            max_photos=self.max_photos,
            camera_index=self.camera_index,
        )
        return saved

    def register_from_image(
        self,
        name: str,
        src: str,
        overwrite: bool = False,
        append: bool = True,
    ) -> int:
        """
        Daftarkan wajah dari file gambar atau folder.

        Args:
            name     : Nama orang.
            src      : Path file gambar atau folder berisi gambar.
            overwrite: Hapus data lama sebelum menyimpan.
            append   : Tambah ke data yang sudah ada.

        Returns:
            Jumlah foto wajah yang tersimpan.
        """
        saved = _dataset_mod.register_from_image(
            name=name,
            src=src,
            overwrite=overwrite,
            append=append,
        )
        return saved

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Latih model dari seluruh dataset yang tersedia.

        Returns:
            dict: {"total_images": int, "total_persons": int, "model_path": str}
        """
        return _trainer_mod.train()

    # ── Deteksi ──────────────────────────────────────────────────────────────

    def detect_camera(self) -> None:
        """Deteksi dan kenali wajah secara real-time dari kamera."""
        _detector_mod.detect_camera(
            threshold=self.threshold,
            camera_index=self.camera_index,
        )

    def detect_image(self, img_path: str, show: bool = True) -> DetectionResult:
        """
        Deteksi dan kenali wajah dari file gambar.

        Args:
            img_path: Path ke file gambar.
            show    : Tampilkan jendela hasil (default True).

        Returns:
            DetectionResult — akses `.faces` untuk list FaceResult.
        """
        return _detector_mod.detect_image(
            img_path=img_path,
            threshold=self.threshold,
            show=show,
        )

    # ── Manajemen Pengguna ───────────────────────────────────────────────────

    def list_users(self) -> list[dict]:
        """
        Dapatkan daftar semua pengguna.

        Returns:
            List of dict: [{"id": int, "name": str, "photos": int}, ...]
        """
        return _users_mod.list_users()

    def delete_user(self, name: str) -> dict:
        """
        Hapus pengguna dan seluruh data fotonya.

        Args:
            name: Nama pengguna (case-insensitive).

        Returns:
            dict: {"id": int, "name": str, "photos_deleted": int}
        """
        return _users_mod.delete_user(name)

    # ── Info ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        users = _labels_mod.load()
        return (
            f"FaceRecog("
            f"users={len(users)}, "
            f"threshold={self.threshold}, "
            f"max_photos={self.max_photos})"
        )


__all__ = ["FaceRecog", "DetectionResult", "FaceResult"]
