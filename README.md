# Sistem Pengenalan Wajah — ecolube.id

Sistem pengenalan wajah berbasis **OpenCV LBPH** (Local Binary Patterns Histograms) yang dapat digunakan melalui **CLI interaktif** maupun **Python API** langsung di kode.

---

## Struktur Proyek

```
ecolube.id/
├── facerecog/
│   ├── __init__.py   ← FaceRecog class (high-level API)
│   ├── config.py     ← path & konstanta global
│   ├── labels.py     ← CRUD labels.json
│   ├── dataset.py    ← registrasi dari kamera & gambar
│   ├── trainer.py    ← training model
│   ├── detector.py   ← deteksi kamera & gambar
│   └── users.py      ← list & hapus pengguna
├── dataset/          ← foto wajah per-orang (auto dibuat)
├── trainer/          ← model hasil training (auto dibuat)
├── labels.json       ← peta ID → nama (auto dibuat)
├── main.py           ← CLI interaktif
├── requirements.txt
└── README.md
```

---

## Instalasi

**1. Clone / buka project**

**2. Install dependensi**

```bash
pip install -r requirements.txt
```

> Pastikan menggunakan `opencv-contrib-python` (bukan `opencv-python`), karena modul `cv2.face.LBPHFaceRecognizer` hanya ada di versi contrib.

---

## Cara Pakai — CLI Interaktif

Jalankan:

```bash
python main.py
```

Akan muncul menu:

```
==============================================
         SISTEM PENGENALAN WAJAH
                ecolube.id
==============================================
  [1]  Register  — Daftarkan wajah via kamera
  [2]  Train     — Latih model
  [3]  Detect    — Deteksi wajah real-time
  [4]  List      — Lihat daftar pengguna
  [5]  Delete    — Hapus pengguna
  [6]  Detect    — Deteksi wajah dari gambar
  [7]  Register  — Daftarkan wajah dari gambar
  [0]  Keluar
==============================================
```

### Alur Dasar

```
[1] atau [7] Register  →  [2] Train  →  [3] atau [6] Detect
```

### Penjelasan Menu

| Menu  | Fungsi                | Keterangan                                        |
| ----- | --------------------- | ------------------------------------------------- |
| `[1]` | Register via kamera   | Ambil 40 foto wajah secara real-time              |
| `[7]` | Register via gambar   | Input path file atau folder gambar                |
| `[2]` | Train model           | Wajib dijalankan setelah register                 |
| `[3]` | Deteksi real-time     | Kamera aktif, tutup dengan `q` / `Esc` / tombol X |
| `[6]` | Deteksi dari gambar   | Input path gambar, hasil tampil di jendela        |
| `[4]` | Lihat daftar pengguna | Tampilkan ID, nama, jumlah foto                   |
| `[5]` | Hapus pengguna        | Hapus data foto + label                           |

---

## Cara Pakai — Python API

Import class `FaceRecog` dari modul `facerecog`:

```python
from facerecog import FaceRecog

fr = FaceRecog()
```

### Konfigurasi (opsional)

```python
fr = FaceRecog(
    threshold=75,      # confidence LBPH, lebih rendah = lebih ketat (default 75)
    max_photos=40,     # jumlah foto per sesi kamera (default 40)
    camera_index=0,    # index kamera (default 0)
)
```

---

### Register Wajah

**Via kamera:**

```python
saved = fr.register_from_camera("Budi")
print(f"{saved} foto tersimpan")
```

**Via file gambar tunggal:**

```python
saved = fr.register_from_image("Budi", "/path/ke/foto_budi.jpg")
```

**Via folder berisi banyak gambar:**

```python
saved = fr.register_from_image("Budi", "/path/ke/folder_foto/")
```

Format gambar yang didukung: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**Opsi overwrite / append:**

```python
# Timpa data lama
fr.register_from_image("Budi", "/foto/", overwrite=True)

# Tambah ke data yang sudah ada (default)
fr.register_from_image("Budi", "/foto/", append=True)
```

---

### Training Model

```python
info = fr.train()
print(info)
# {"total_images": 80, "total_persons": 2, "model_path": "/path/trainer/trainer.yml"}
```

> Wajib dijalankan ulang setiap kali ada penambahan atau penghapusan data.

---

### Deteksi Wajah

**Real-time dari kamera:**

```python
fr.detect_camera()
# Tutup dengan q, Esc, atau klik tombol X pada window
```

**Dari file gambar (dengan tampilan window):**

```python
result = fr.detect_image("foto.jpg", show=True)
```

**Dari file gambar (tanpa tampilan window, hanya data):**

```python
result = fr.detect_image("foto.jpg", show=False)

print(f"Total wajah: {result.total_faces}")

for face in result.faces:
    print(face.name)        # nama orang / "Tidak Dikenal"
    print(face.recognized)  # True / False
    print(face.score)       # keyakinan dalam persen (0-100)
    print(face.confidence)  # raw LBPH confidence (lebih rendah = lebih yakin)
    print(face.x, face.y, face.w, face.h)  # posisi & ukuran bounding box
```

**Contoh: cek apakah wajah dikenali**

```python
result = fr.detect_image("foto.jpg", show=False)

for face in result.faces:
    if face.recognized:
        print(f"Dikenali: {face.name} ({face.score}%)")
    else:
        print("Wajah tidak dikenal")
```

---

### Manajemen Pengguna

**Lihat daftar pengguna:**

```python
users = fr.list_users()
for u in users:
    print(u["id"], u["name"], u["photos"])
# 1  Budi   40
# 2  Siti   35
```

**Hapus pengguna:**

```python
info = fr.delete_user("Budi")
print(info)
# {"id": 1, "name": "Budi", "photos_deleted": 40}
```

> Setelah hapus, jalankan `fr.train()` ulang agar model diperbarui.

---

### Info Instance

```python
print(fr)
# FaceRecog(users=2, threshold=75, max_photos=40)
```

---

## Contoh Lengkap

```python
from facerecog import FaceRecog

fr = FaceRecog(threshold=70)

# 1. Daftarkan dua orang dari folder foto
fr.register_from_image("Budi", "./foto/budi/")
fr.register_from_image("Siti", "./foto/siti/")

# 2. Train model
info = fr.train()
print(f"Training selesai: {info['total_images']} gambar, {info['total_persons']} orang")

# 3. Cek siapa yang ada di foto
result = fr.detect_image("./test/foto_group.jpg", show=True)
for face in result.faces:
    status = f"{face.name} ({face.score}%)" if face.recognized else "Tidak Dikenal"
    print(f"Wajah di ({face.x},{face.y}): {status}")

# 4. Deteksi real-time
fr.detect_camera()
```

---

## Catatan

- Model disimpan di `trainer/trainer.yml` dan labels di `labels.json` — keduanya otomatis dibuat.
- Semakin banyak foto dataset, semakin akurat hasil pengenalan.
- Jika kamera tidak terdeteksi, coba ubah `camera_index=1` atau `camera_index=2`.
- Gunakan foto wajah yang jelas, pencahayaan cukup, dan menghadap depan untuk hasil terbaik.
