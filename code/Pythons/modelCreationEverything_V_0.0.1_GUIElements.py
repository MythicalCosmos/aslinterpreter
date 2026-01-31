import os
import re
import time
import threading
import zipfile
from pathlib import Path
from tkinter import (
    Tk, Label, Button, Listbox, Entry, StringVar, Frame, Scale,
    HORIZONTAL, filedialog, messagebox, ttk, OptionMenu, IntVar
)
from PIL import Image, ImageTk
import cv2
#import tensorflow as tf
#from tensorflow import keras

BASE_DATA_DIR = Path(r"~\Software Dev 2\Software Dev\gesture_data")
DEFAULT_MODEL_BASENAME = "gesture_recognizer.task"
CAMERA_INDEX = 0

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_next_version_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    version_re = re.compile(r"version_(\d+)$")
    highest = 0
    for d in base_dir.iterdir():
        if d.is_dir() and version_re.match(d.name):
            num = int(version_re.match(d.name).group(1))
            highest = max(highest, num)
    next_num = highest + 1
    version_dir = base_dir / f"version_{next_num}"
    images_dir = version_dir / "images"
    export_dir = version_dir / "export"

    images_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    return version_dir



def next_model_filename(output_dir: Path, base_name: str = DEFAULT_MODEL_BASENAME) -> Path:
    base = Path(base_name)
    stem = base.stem
    suffix = base.suffix or ".task"
    pattern = re.compile(re.escape(stem) + r"(?:_(\d+))?" + re.escape(suffix) + r"$")
    highest = 0
    for f in output_dir.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                n = int(m.group(1) or 1)
                highest = max(highest, n)
    if highest == 0:
        target = output_dir / f"{stem}{suffix}"
    else:
        target = output_dir / f"{stem}_{highest+1}{suffix}"
    return target

def write_basic_metadata(meta_path, label_list):
    """Creates a simple metadata text file at the given path."""
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("description: Gesture recognizer metadata\n")
        f.write("labels:\n")
        for label in label_list:
            f.write(f"  - {label}\n")
    return meta_path

class GestureTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Trainer (TensorFlow Edition)")

        self.version_dir = get_next_version_dir(BASE_DATA_DIR)
        self.status_text = StringVar(value=f"Version folder: {self.version_dir}")

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open webcam.")
            raise RuntimeError("Could not open webcam")
        self.frame = None
        self.preview_image = None
        self.running = True

        self.current_label = None
        self.current_hand = "both"
        self.global_image_counter = 0

        self.setup_gui()

    # ---------------- GUI SETUP ----------------
    def setup_gui(self):
        left_frame = Frame(self.root)
        left_frame.grid(row=0, column=0, padx=8, pady=8, sticky="n")

        right_frame = Frame(self.root)
        right_frame.grid(row=0, column=1, padx=8, pady=8, sticky="n")

        self.preview_label = Label(right_frame)
        self.preview_label.pack()

        status_bar = Label(self.root, textvariable=self.status_text, anchor="w")
        status_bar.grid(row=2, column=0, columnspan=2, sticky="we", padx=8, pady=(4, 8))

        Label(left_frame, text="Gestures:").pack(anchor="w")
        self.gesture_list = Listbox(left_frame, width=28, height=8)
        self.gesture_list.pack()

        lbl_frame = Frame(left_frame)
        lbl_frame.pack(fill="x", pady=(6, 4))
        Label(lbl_frame, text="New gesture name:").grid(row=0, column=0, sticky="w")
        self.new_label_entry = Entry(lbl_frame)
        self.new_label_entry.grid(row=0, column=1, padx=(6, 0), sticky="we")
        lbl_frame.columnconfigure(1, weight=1)

        
        Button(lbl_frame, text="Add", command=self.add_label).grid(row=0, column=3, padx=(6, 0))

        # Capture controls
        cap_frame = Frame(left_frame, pady=6)

        Button(cap_frame, text="Open Version Folder", command=self.open_version_folder).grid(row=3, column=0, columnspan=3, pady=(8, 0), sticky="we")

        rm_frame = Frame(left_frame, pady=6)
        rm_frame.pack(fill="x")
        Button(rm_frame, text="Delete Selected Label", command=self.delete_selected_label).pack(fill="x")

    # ---------------- CAPTURE ----------------
    def add_label(self):
        name = self.new_label_entry.get().strip()
        if not name:
            messagebox.showwarning("No name", "Please enter a gesture name.")
            return
        label = name.replace(" ", "_")
        hand = self.hand_var.get()
        entry_name = f"{hand}__{label}"
        existing = [self.gesture_list.get(i) for i in range(self.gesture_list.size())]
        if entry_name in existing:
            messagebox.showinfo("Exists", "That label already exists.")
            return
        self.gesture_list.insert("end", entry_name)
        (self.version_dir / "images" / entry_name).mkdir(parents=True, exist_ok=True)
        self.new_label_entry.delete(0, "end")
        self.status_text.set(f"Added label {entry_name}")

    def delete_selected_label(self):
        sel = self.gesture_list.curselection()
        if not sel:
            return
        val = self.gesture_list.get(sel[0])
        if messagebox.askyesno("Confirm", f"Delete label entry '{val}' (files kept)?"):
            self.gesture_list.delete(sel[0])
            self.status_text.set(f"Removed {val}")

    def open_version_folder(self):
        os.startfile(self.version_dir)
    
    def get_selected_label(self):
        sel = self.gesture_list.curselection()
        return self.gesture_list.get(sel[0]) if sel else None

    # ---------------- TRAIN / EXPORT ----------------
    def train_and_export(self):
        print("trainbutton")

    def close(self):
        self.running = False
        time.sleep(0.05)
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.quit()


def main():
    root = Tk()
    gui = GestureTrainerGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
