"""
facerecog — Face Recognition Module
High-level API: all features available through the FaceRecog class.

Quick start:

    from facerecog import FaceRecog

    fr = FaceRecog()

    # Register via camera
    fr.register_from_camera("Alice")

    # Register via image file or folder
    fr.register_from_image("Alice", "/path/to/photos/")

    # Train the model
    fr.train()

    # Real-time detection
    fr.detect_camera()

    # Detect from image
    result = fr.detect_image("/path/photo.jpg", show=True)
    for face in result.faces:
        print(face.name, face.score)

    # List users
    users = fr.list_users()

    # Delete user
    fr.delete_user("Alice")

For a full interactive CLI demo, run:
    python facerecog/example.py
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
    Main interface for the face recognition module.

    All operations (register, train, detect, manage users) are
    available as methods on this class.
    """

    # ── Konfigurasi ──────────────────────────────────────────────────────────

    def __init__(
        self,
        threshold: int = CONFIDENCE_THRESHOLD,
        max_photos: int = MAX_PHOTOS,
        camera_index: int = 0,
        app_name: str = "Face Recognition",
    ):
        """
        Args:
            threshold   : LBPH confidence limit (default 75). Lower = stricter.
            max_photos  : Photos per camera registration session (default 40).
            camera_index: Camera index to use (default 0).
            app_name    : Application name shown in OpenCV window titles.
        """
        self.threshold    = threshold
        self.max_photos   = max_photos
        self.camera_index = camera_index
        self.app_name     = app_name

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
            app_name=self.app_name,
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
        """Detect and recognize faces in real-time from camera."""
        _detector_mod.detect_camera(
            threshold=self.threshold,
            camera_index=self.camera_index,
            app_name=self.app_name,
        )

    def detect_image(self, img_path: str, show: bool = True) -> DetectionResult:
        """
        Detect and recognize faces from an image file.

        Args:
            img_path: Path to image file.
            show    : Show result window (default True).

        Returns:
            DetectionResult — access `.faces` for list of FaceResult.
        """
        return _detector_mod.detect_image(
            img_path=img_path,
            threshold=self.threshold,
            show=show,
            app_name=self.app_name,
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
            f"app_name='{self.app_name}', "
            f"users={len(users)}, "
            f"threshold={self.threshold}, "
            f"max_photos={self.max_photos})"
        )


__all__ = ["FaceRecog", "DetectionResult", "FaceResult"]
