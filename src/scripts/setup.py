#!/usr/bin/env python3
import sys
import urllib.request
import urllib.error
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent.parent
DEPLOY_DIR  = _REPO_ROOT / "src/deploy"

MODELS = [
    {
        "name": "hand_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        ),
        "dest": DEPLOY_DIR / "hand_landmarker.task",
        "required": True,
        "description": "MediaPipe hand landmark detection model (~29MB)",
    },
]


def download(url: str, dest: Path, description: str) -> bool:
    print(f"\nDownloading: {description}")
    print(f"  Source : {url}")
    print(f"  Dest   : {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        def progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(block_num * block_size / total_size * 100, 100)
                bar_len = 40
                filled = int(bar_len * pct / 100)
                bar = "#" * filled + "-" * (bar_len - filled)
                print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print(f"\n  [OK] Saved ({dest.stat().st_size / 1_048_576:.1f} MB)")
        return True
    except urllib.error.URLError as e:
        print(f"\n  [FAIL] Download failed: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("  ASL Interpreter — Setup")
    print("=" * 60)

    all_ok = True
    for model in MODELS:
        dest = model["dest"]
        if dest.exists():
            print(f"\n[OK] {model['name']} already exists ({dest.stat().st_size / 1_048_576:.1f} MB)")
            continue
        ok = download(model["url"], dest, model["description"])
        if not ok and model["required"]:
            all_ok = False
            print(f"  ERROR: {model['name']} is required. Check your internet connection.")

    print("\n" + "=" * 60)
    if all_ok:
        print("  Setup complete! You can now launch the application.")
        print("  Run: python src/app/src/main.py")
    else:
        print("  Setup incomplete. Some required files could not be downloaded.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()