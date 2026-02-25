"""
facerecog/detector.py
Deteksi & pengenalan wajah — dari kamera real-time atau dari file gambar.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import cv2

from .config import CASCADE_PATH, MODEL_PATH, CONFIDENCE_THRESHOLD
from . import labels as lbl


# ─── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class FaceResult:
    """Hasil deteksi satu wajah."""
    x: int
    y: int
    w: int
    h: int
    user_id: Optional[int]       # None jika tidak dikenali
    name: str                    # "Tidak Dikenal" jika tidak dikenali
    confidence: float            # raw LBPH confidence (lebih rendah = lebih yakin)
    recognized: bool             # True jika confidence < threshold

    @property
    def score(self) -> int:
        """Keyakinan dalam persen (0–100). Hanya bermakna jika recognized=True."""
        return 100 - int(self.confidence)


@dataclass
class DetectionResult:
    """Hasil deteksi satu gambar."""
    image_path: Optional[str]
    total_faces: int
    faces: list[FaceResult] = field(default_factory=list)


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model belum ada. Jalankan train() terlebih dahulu.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    return recognizer


def _draw_result(frame, result: FaceResult):
    color = (0, 220, 0) if result.recognized else (0, 0, 220)
    label = f"{result.name}  {result.score}%".strip() if result.recognized else result.name
    cv2.rectangle(frame, (result.x, result.y),
                  (result.x + result.w, result.y + result.h), color, 2)
    cv2.putText(frame, label, (result.x, result.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def _wait_close(window_title: str):
    """Loop tunggu sampai window ditutup via X, q, atau Esc."""
    while True:
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_camera(
    threshold: int = CONFIDENCE_THRESHOLD,
    camera_index: int = 0,
) -> None:
    """
    Deteksi dan kenali wajah secara real-time dari kamera.

    Args:
        threshold   : LBPH confidence < threshold = dikenali.
        camera_index: Index kamera (default 0).

    Raises:
        RuntimeError: Jika model belum ada atau kamera tidak bisa dibuka.
    """
    recognizer   = _load_model()
    labels       = lbl.load()
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka kamera (index {camera_index}).")

    WIN = "Deteksi Wajah — ecolube.id"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
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
            lid, conf = recognizer.predict(gray[y:y + h, x:x + w])
            recognized = conf < threshold
            result = FaceResult(
                x=x, y=y, w=w, h=h,
                user_id=lid if recognized else None,
                name=labels.get(str(lid), "?") if recognized else "Tidak Dikenal",
                confidence=conf,
                recognized=recognized,
            )
            _draw_result(frame, result)

        cv2.putText(frame, f"Terdaftar: {len(labels)} orang", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
        cv2.putText(frame, "Tekan 'q' / Esc untuk keluar", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_image(
    img_path: str,
    threshold: int = CONFIDENCE_THRESHOLD,
    show: bool = True,
) -> DetectionResult:
    """
    Deteksi dan kenali wajah dari file gambar.

    Args:
        img_path  : Path ke file gambar.
        threshold : LBPH confidence < threshold = dikenali.
        show      : Tampilkan jendela hasil jika True.

    Returns:
        DetectionResult berisi daftar FaceResult.

    Raises:
        ValueError  : Jika file tidak ditemukan atau gagal dibaca.
        RuntimeError: Jika model belum ada.
    """
    if not os.path.exists(img_path):
        raise ValueError(f"File tidak ditemukan: {img_path}")

    recognizer   = _load_model()
    labels       = lbl.load()
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    frame = cv2.imread(img_path)
    if frame is None:
        raise ValueError(f"Gagal membaca gambar: {img_path}")

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    results: list[FaceResult] = []

    if len(faces) == 0:
        if show:
            WIN = "Deteksi Wajah dari Gambar — ecolube.id"
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.putText(frame, "Tidak ada wajah terdeteksi", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
            cv2.putText(frame, "Tekan 'q' / Esc / X untuk tutup",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(WIN, frame)
            _wait_close(WIN)
        return DetectionResult(image_path=img_path, total_faces=0, faces=[])

    for (x, y, w, h) in faces:
        lid, conf = recognizer.predict(gray[y:y + h, x:x + w])
        recognized = conf < threshold
        r = FaceResult(
            x=x, y=y, w=w, h=h,
            user_id=lid if recognized else None,
            name=labels.get(str(lid), "?") if recognized else "Tidak Dikenal",
            confidence=conf,
            recognized=recognized,
        )
        results.append(r)
        if show:
            _draw_result(frame, r)

    if show:
        WIN = "Deteksi Wajah dari Gambar — ecolube.id"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.putText(frame, f"Terdaftar: {len(labels)} orang", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
        cv2.putText(frame, "Tekan 'q' / Esc / X untuk tutup",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(WIN, frame)
        _wait_close(WIN)

    return DetectionResult(
        image_path=img_path,
        total_faces=len(results),
        faces=results,
    )
