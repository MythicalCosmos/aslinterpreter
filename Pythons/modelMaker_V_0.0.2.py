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

BASE_DATA_DIR = Path(r"C:\Users\360mc\Stuff\Downloads\Software Dev 2\Software Dev\gesture_data")
DEFAULT_MODEL_BASENAME = "gesture_recognizer.task"
CAMERA_INDEX = 0

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_next_version_dir(base_dir: Path) -> Path:
    """Creates a new version_X directory inside gesture_data, 
    with subfolders for images and export."""
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
        self.update_preview()

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

        hand_options = ["left", "right", "both"]
        self.hand_var = StringVar(value="both")
        OptionMenu(lbl_frame, self.hand_var, *hand_options).grid(row=0, column=2, padx=(6, 0))
        Button(lbl_frame, text="Add", command=self.add_label).grid(row=0, column=3, padx=(6, 0))

        # Capture controls
        cap_frame = Frame(left_frame, pady=6)
        cap_frame.pack(fill="x")
        Label(cap_frame, text="Burst count:").grid(row=0, column=0, sticky="w")
        self.burst_count_var = IntVar(value=5)
        Scale(cap_frame, from_=1, to=30, orient=HORIZONTAL, variable=self.burst_count_var).grid(row=0, column=1, columnspan=3, sticky="we")

        Label(cap_frame, text="Burst interval (ms):").grid(row=1, column=0, sticky="w")
        self.burst_interval_var = IntVar(value=150)
        Scale(cap_frame, from_=50, to=1000, orient=HORIZONTAL, variable=self.burst_interval_var).grid(row=1, column=1, columnspan=3, sticky="we")

        Button(cap_frame, text="Capture Single", command=self.capture_single).grid(row=2, column=0, sticky="we")
        Button(cap_frame, text="Capture Burst", command=self.capture_burst_thread).grid(row=2, column=1, sticky="we")
        Button(cap_frame, text="Auto-Capture", command=self.auto_capture_prompt).grid(row=2, column=2, sticky="we")
        Button(cap_frame, text="Open Version Folder", command=self.open_version_folder).grid(row=3, column=0, columnspan=3, pady=(8, 0), sticky="we")

        # Train/export
        train_frame = Frame(left_frame, pady=6)
        train_frame.pack(fill="x")
        Button(train_frame, text="Train & Export (.task)", command=self.train_and_export).pack(fill="x")
        self.progress = ttk.Progressbar(train_frame, length=200, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 0))

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

    def update_preview(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.frame = frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((640, 480))
            self.preview_image = ImageTk.PhotoImage(img_pil)
            self.preview_label.configure(image=self.preview_image)
        self.root.after(30, self.update_preview)

    def get_selected_label(self):
        sel = self.gesture_list.curselection()
        return self.gesture_list.get(sel[0]) if sel else None

    def _ensure_label_dir(self, entry_name):
        img_root = self.version_dir / "images"
        sub = img_root / entry_name
        sub.mkdir(parents=True, exist_ok=True)
        return sub


    def _next_image_path(self, label_dir: Path, label_name: str) -> Path:
        pattern = re.compile(re.escape(label_name) + r"_image_(\d+)\.jpg$")
        highest = 0
        for f in label_dir.iterdir():
            if f.is_file():
                m = pattern.match(f.name)
                if m:
                    highest = max(highest, int(m.group(1)))
        filename = f"{label_name}_image_{highest+1:04d}.jpg"
        return label_dir / filename

    def capture_single(self):
        sel = self.get_selected_label()
        if not sel:
            messagebox.showinfo("Select", "Select a gesture first.")
            return
        if self.frame is None:
            return
        hand, label = sel.split("__", 1)
        label_name = f"{hand}_hand_{label}"
        path = self._next_image_path(self._ensure_label_dir(sel), label_name)
        cv2.imwrite(str(path), self.frame)
        self.status_text.set(f"Saved {path.name}")

    def capture_burst_thread(self):
        threading.Thread(target=self.capture_burst, daemon=True).start()

    def capture_burst(self):
        sel = self.get_selected_label()
        if not sel:
            messagebox.showinfo("Select", "Select a gesture first.")
            return
        count = self.burst_count_var.get()
        interval = self.burst_interval_var.get() / 1000.0
        hand, label = sel.split("__", 1)
        label_name = f"{hand}_hand_{label}"
        label_dir = self._ensure_label_dir(sel)
        for i in range(count):
            if self.frame is not None:
                path = self._next_image_path(label_dir, label_name)
                cv2.imwrite(str(path), self.frame)
                self.status_text.set(f"Burst {i+1}/{count}")
            time.sleep(interval)
        self.status_text.set(f"Burst done: {count} images saved.")

    def auto_capture_prompt(self):
        messagebox.showinfo("Auto-Capture", "Capturing 20 frames at 100 ms intervals.")
        threading.Thread(target=self.auto_capture_worker, daemon=True).start()

    def auto_capture_worker(self):
        sel = self.get_selected_label()
        if not sel:
            return
        hand, label = sel.split("__", 1)
        label_name = f"{hand}_hand_{label}"
        label_dir = self._ensure_label_dir(sel)
        for i in range(20):
            if self.frame is not None:
                path = self._next_image_path(label_dir, label_name)
                cv2.imwrite(str(path), self.frame)
                self.status_text.set(f"Auto {i+1}/20")
            time.sleep(0.1)
        self.status_text.set("Auto-capture complete.")

    # ---------------- TRAIN / EXPORT ----------------
    def train_and_export(self):
        print("trainbutton")
        '''
        subfolders = [d for d in self.version_dir.iterdir() if d.is_dir()]
        if not subfolders:
            messagebox.showinfo("No data", "No labeled gesture folders found.")
            return
        threading.Thread(target=self._train_worker, args=(self.version_dir,), daemon=True).start()

    def _train_worker(self, dataset_root: Path):
        try:
            self.progress['value'] = 0
            self.status_text.set("Loading dataset...")

            img_size = (224, 224)
            batch_size = 16

            images_root = dataset_root / "images"

            train_ds = tf.keras.utils.image_dataset_from_directory(
                images_root,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=img_size,
                batch_size=batch_size
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                images_root,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=img_size,
                batch_size=batch_size
            )

            class_names = train_ds.class_names
            self.status_text.set(f"Classes: {class_names}")

            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            model = keras.Sequential([
                keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
                keras.layers.Conv2D(32, 3, activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(64, 3, activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(128, 3, activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(len(class_names), activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            self.status_text.set("Training model...")
            self.progress['value'] = 30
            model.fit(train_ds, validation_data=val_ds, epochs=10)
            self.progress['value'] = 70

            loss, acc = model.evaluate(val_ds)
            self.status_text.set(f"Validation accuracy: {acc:.3f}")
            self.progress['value'] = 80

            export_dir = self.version_dir / "export"
            export_dir.mkdir(parents=True, exist_ok=True)
            out_model_path = next_model_filename(export_dir)
            tflite_path = str(out_model_path.with_suffix('.tflite'))




            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            self.status_text.set("Packaging .task archive...")

            meta_path = export_dir / "metadata.json"
            labels_path = export_dir / "labels.txt"
            write_basic_metadata(meta_path, class_names)


            # Create a label file that MediaPipe expects
            with open(labels_path, "w", encoding="utf-8") as f:
                for name in class_names:
                    f.write(name + "\n")

            # Create a proper .task ZIP archive
            with zipfile.ZipFile(out_model_path, "w", zipfile.ZIP_STORED) as zf:
                zf.write(tflite_path, arcname="model.tflite")
                zf.write(meta_path, arcname="metadata.json")
                zf.write(labels_path, arcname="labels.txt")



            self.progress['value'] = 100
            messagebox.showinfo("Done", f"Model exported to:\n{out_model_path}")
            self.status_text.set(f"Exported model to {out_model_path.name}")
            time.sleep(1)
            self.progress['value'] = 0

        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self.status_text.set("Training failed.")
            '''

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
