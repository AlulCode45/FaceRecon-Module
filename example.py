"""
example.py — Interactive CLI demo for the facerecog module.

Run from the repo root:
    python example.py

All data (dataset/, trainer/, labels.json) will be created
in your current working directory.
"""
import sys
import os
from facerecog import FaceRecog
from facerecog.config import CASCADE_PATH

fr = FaceRecog()


# ─── Menu Handlers ────────────────────────────────────────────────────────────

def menu_register_camera():
    name = input("\nEnter name: ").strip()
    if not name:
        print("[!] Name cannot be empty.")
        return
    try:
        saved = fr.register_from_camera(name)
        if saved > 0:
            print(f"\n[✓] {saved} photos saved for '{name}'.")
            print("[*] Run Train (menu 2) to activate this data.")
        else:
            print("\n[!] Registration cancelled — no photos saved.")
    except (ValueError, RuntimeError) as e:
        print(f"[!] {e}")


def menu_register_image():
    name = input("\nEnter name: ").strip()
    if not name:
        print("[!] Name cannot be empty.")
        return
    src = input("Enter image file path or folder: ").strip()
    try:
        saved = fr.register_from_image(name, src)
        print(f"\n[✓] {saved} face photos saved for '{name}'.")
        print("[*] Run Train (menu 2) to activate this data.")
    except (ValueError, RuntimeError) as e:
        print(f"[!] {e}")


def menu_train():
    try:
        info = fr.train()
        print(f"\n[✓] Training complete — {info['total_images']} images, "
              f"{info['total_persons']} people.")
        print(f"[*] Model saved: {info['model_path']}")
    except RuntimeError as e:
        print(f"\n[!] {e}")


def menu_detect_camera():
    try:
        print("\n[*] Starting camera detection… Press 'q' / Esc / X to quit.\n")
        fr.detect_camera()
        print("[*] Camera detection stopped.")
    except RuntimeError as e:
        print(f"\n[!] {e}")


def menu_detect_image():
    img_path = input("\nEnter image file path: ").strip()
    try:
        result = fr.detect_image(img_path, show=True)
        if result.total_faces == 0:
            print("[!] No face detected in image.")
        else:
            print(f"\n[✓] {result.total_faces} face(s) detected:")
            for i, face in enumerate(result.faces, 1):
                if face.recognized:
                    print(f"  Face {i}: {face.name}  (confidence: {face.score}%)")
                else:
                    print(f"  Face {i}: Unknown")
        print("[*] Detection window closed.")
    except (ValueError, RuntimeError) as e:
        print(f"\n[!] {e}")


def menu_list_users():
    users = fr.list_users()
    if not users:
        print("\n[!] No users registered yet.")
        return
    print("\n" + "=" * 46)
    print(f"{'REGISTERED USERS':^46}")
    print("=" * 46)
    print(f"  {'ID':<6} {'Name':<24} {'Photos'}")
    print("-" * 46)
    for u in users:
        print(f"  {u['id']:<6} {u['name']:<24} {u['photos']}")
    print("=" * 46)
    print(f"  Total: {len(users)} user(s)")


def menu_delete_user():
    menu_list_users()
    name = input("\nEnter name to delete: ").strip()
    confirm = input(f"Delete '{name}'? Photo data will also be removed. (y/n): ").strip().lower()
    if confirm != "y":
        print("[*] Deletion cancelled.")
        return
    try:
        info = fr.delete_user(name)
        print(f"[✓] '{info['name']}' deleted ({info['photos_deleted']} photos removed).")
        print("[*] Re-train the model for changes to take effect.")
    except ValueError as e:
        print(f"[!] {e}")


# ─── Menu Display ─────────────────────────────────────────────────────────────

def print_menu():
    print("\n" + "=" * 46)
    print(f"{'FACE RECOGNITION SYSTEM':^46}")
    print("=" * 46)
    print("  [1]  Register  — Register face via camera")
    print("  [2]  Train     — Train the model")
    print("  [3]  Detect    — Real-time face detection")
    print("  [4]  List      — List registered users")
    print("  [5]  Delete    — Delete a user")
    print("  [6]  Detect    — Detect face from image")
    print("  [7]  Register  — Register face from image")
    print("  [0]  Exit")
    print("=" * 46)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CASCADE_PATH):
        print(f"[!] Haar Cascade not found:\n    {CASCADE_PATH}")
        sys.exit(1)

    print("\n[*] Face Recognition System ready.")

    MENU = {
        "1": menu_register_camera,
        "2": menu_train,
        "3": menu_detect_camera,
        "4": menu_list_users,
        "5": menu_delete_user,
        "6": menu_detect_image,
        "7": menu_register_image,
    }

    while True:
        print_menu()
        choice = input("Select menu: ").strip()
        if choice == "0":
            print("\n[*] Goodbye!\n")
            sys.exit(0)
        elif choice in MENU:
            MENU[choice]()
        else:
            print("[!] Invalid choice.")


if __name__ == "__main__":
    main()
