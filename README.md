# facerecog — Face Recognition Module

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/opencv--contrib--python-4.8%2B-green)](https://pypi.org/project/opencv-contrib-python/)

A lightweight face recognition module based on **OpenCV LBPH** (Local Binary Patterns Histograms).  
Use it via the **interactive CLI** or embed it directly as a **Python API** in your own project.

---

## Features

- Register faces from **camera** or **image files / folders**
- Train the LBPH recognition model
- Detect & recognize faces **real-time** from camera or from **image files**
- Manage registered users (list, delete)
- Clean Python API — one class, zero boilerplate
- Customizable window titles via `app_name`

---

## Requirements

- Python 3.10+
- `opencv-contrib-python >= 4.8.0`
- `numpy >= 1.24.0`

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/AlulCode45/FaceRecon-Module.git
cd FaceRecon-Module
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> **Important:** Use `opencv-contrib-python`, not `opencv-python`.  
> The `cv2.face.LBPHFaceRecognizer` module is only available in the contrib build.

---

## Project Structure

When you use this module, it will automatically create the following in your **working directory**:

```
your-project/
├── facerecog/          ← this module (clone here)
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── detector.py
│   ├── labels.py
│   ├── trainer.py
│   ├── users.py
│   ├── example.py
│   ├── requirements.txt
│   └── README.md
├── dataset/            ← auto created: face photos per person
├── trainer/            ← auto created: trained model output
└── labels.json         ← auto created: ID → name mapping
```

---

## Usage — Interactive CLI

Run the built-in CLI demo:

```bash
python facerecog/example.py
```

Menu:

```
==============================================
         FACE RECOGNITION SYSTEM
==============================================
  [1]  Register  — Register face via camera
  [2]  Train     — Train the model
  [3]  Detect    — Real-time face detection
  [4]  List      — List registered users
  [5]  Delete    — Delete a user
  [6]  Detect    — Detect face from image
  [7]  Register  — Register face from image
  [0]  Exit
==============================================
```

### Basic flow

```
[1] or [7] Register  →  [2] Train  →  [3] or [6] Detect
```

---

## Usage — Python API

```python
from facerecog import FaceRecog

fr = FaceRecog()
```

### Configuration (optional)

```python
fr = FaceRecog(
    threshold=75,                # LBPH confidence: lower = stricter match (default 75)
    max_photos=40,               # photos captured per camera session (default 40)
    camera_index=0,              # camera device index (default 0)
    app_name="My App",           # label in OpenCV window titles (default "Face Recognition")
)
```

---

### Register Faces

**Via camera:**

```python
saved = fr.register_from_camera("Alice")
print(f"{saved} photos saved")
```

**Via single image file:**

```python
saved = fr.register_from_image("Alice", "/path/to/alice.jpg")
```

**Via folder containing multiple images:**

```python
saved = fr.register_from_image("Alice", "/path/to/alice_photos/")
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**Overwrite / append:**

```python
# Replace existing data
fr.register_from_image("Alice", "/photos/", overwrite=True)

# Add to existing data (default)
fr.register_from_image("Alice", "/photos/", append=True)
```

---

### Train the Model

```python
info = fr.train()
print(info)
# {"total_images": 80, "total_persons": 2, "model_path": "trainer/trainer.yml"}
```

> **Must be re-run** whenever faces are added or deleted.

---

### Detect Faces

**Real-time from camera:**

```python
fr.detect_camera()
# Close with q, Esc, or click X on the window
```

**From image file (with result window):**

```python
result = fr.detect_image("photo.jpg", show=True)
```

**From image file (data only, no window):**

```python
result = fr.detect_image("photo.jpg", show=False)

print(f"Faces found: {result.total_faces}")

for face in result.faces:
    print(face.name)        # person name / "Unknown"
    print(face.recognized)  # True / False
    print(face.score)       # confidence percentage (0–100)
    print(face.confidence)  # raw LBPH value (lower = more confident)
    print(face.x, face.y, face.w, face.h)  # bounding box
```

**Check if a face is recognized:**

```python
result = fr.detect_image("photo.jpg", show=False)

for face in result.faces:
    if face.recognized:
        print(f"Recognized: {face.name} ({face.score}%)")
    else:
        print("Unknown face")
```

---

### User Management

**List all users:**

```python
users = fr.list_users()
for u in users:
    print(u["id"], u["name"], u["photos"])
# 1  Alice   40
# 2  Bob     35
```

**Delete a user:**

```python
info = fr.delete_user("Alice")
print(info)
# {"id": 1, "name": "Alice", "photos_deleted": 40}
```

> After deleting, call `fr.train()` again to update the model.

---

### Instance Info

```python
print(fr)
# FaceRecog(app_name='Face Recognition', users=2, threshold=75, max_photos=40)
```

---

## Full Example

```python
from facerecog import FaceRecog

fr = FaceRecog(threshold=70, app_name="Security System")

# 1. Register people from photo folders
fr.register_from_image("Alice", "./photos/alice/")
fr.register_from_image("Bob",   "./photos/bob/")

# 2. Train the model
info = fr.train()
print(f"Done: {info['total_images']} images, {info['total_persons']} people")

# 3. Detect faces in a group photo
result = fr.detect_image("group.jpg", show=True)
for face in result.faces:
    status = f"{face.name} ({face.score}%)" if face.recognized else "Unknown"
    print(f"Face at ({face.x},{face.y}): {status}")

# 4. Real-time detection
fr.detect_camera()
```

---

## Notes

- `dataset/`, `trainer/`, and `labels.json` are created in your **current working directory** — not inside the module folder.
- More training photos = better accuracy.
- If the camera is not detected, try `camera_index=1` or `camera_index=2`.
- Use clear, well-lit, front-facing photos for best results.
- The `app_name` parameter sets the title of all OpenCV windows — useful when integrating into a larger app.

---

## License

MIT
