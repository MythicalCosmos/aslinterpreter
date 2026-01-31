#!/usr/bin/env python3

import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
from datetime import datetime
from mediapipe_model_maker import gesture_recognizer
import shutil
import tempfile

DATASET_PATH = os.path.expanduser("~/gesture_data")
os.makedirs(DATASET_PATH, exist_ok=True)


class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dataset Manager")
        self.root.geometry("1200x950")
        self.root.resizable(False, False)

        self.cap = None
        self.detect_camera()
        self.capturing = False
        self.trained_models = []
        self.checkbox_vars = {}
        self.model_checkbox_vars = {}

        self.build_gui()
        self.refresh_gesture_list()
        self.refresh_model_list()

        if self.cap:
            self.update_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- Camera ----------------
    def detect_camera(self):
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    print(f"Camera found at index {i}")
                    return
            cap.release()
        print("No working webcam detected.")
        self.cap = None

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2.resize(rgb, (600, 450)))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # Automatic capture
                if self.capturing:
                    selected_paths = [p for p, var in self.checkbox_vars.items() if var.get()]
                    for path in selected_paths:
                        os.makedirs(path, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filepath = os.path.join(path, f"{timestamp}.jpg")
                        cv2.imwrite(filepath, frame)
                    if selected_paths:
                        self.refresh_gesture_list()

        self.root.after(50, self.update_frame)

    # ---------------- GUI ----------------
    def build_gui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=5)

        # Camera display
        self.video_label = ttk.Label(top_frame)
        self.video_label.pack()

        # Controls
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(pady=5)

        ttk.Label(control_frame, text="Gesture Name:").grid(row=0, column=0, padx=5)
        self.entry_gesture = ttk.Entry(control_frame, width=20)
        self.entry_gesture.grid(row=0, column=1, padx=5)

        ttk.Button(control_frame, text="Add Gesture", command=self.add_gesture).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="📸 Capture", command=self.toggle_capture).grid(row=0, column=3, padx=5)

        # Gesture Table
        table_frame = ttk.LabelFrame(self.root, text="Gestures")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.tree = ttk.Treeview(table_frame, columns=("check", "count", "path"), show="headings")
        self.tree.heading("check", text="Select")
        self.tree.heading("count", text="Image Count")
        self.tree.heading("path", text="Path")
        self.tree.column("check", width=60)
        self.tree.column("count", width=100)
        self.tree.column("path", width=900)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Button-1>", self.on_tree_click)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=10)

        # Trained Models Table
        model_frame = ttk.LabelFrame(self.root, text="Trained Models")
        model_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.model_tree = ttk.Treeview(model_frame, columns=("check", "path"), show="headings")
        self.model_tree.heading("check", text="Select")
        self.model_tree.heading("path", text="Model Path")
        self.model_tree.column("check", width=60)
        self.model_tree.column("path", width=900)
        self.model_tree.pack(fill="both", expand=True)
        self.model_tree.bind("<Button-1>", self.on_model_tree_click)

        # Bottom Buttons (full row)
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="🔄 Refresh", command=self.refresh_gesture_list).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🧠 Train Selected", command=self.train_selected).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="📦 Export Selected Model", command=self.export_selected_models).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🗑️ Delete Gesture", command=self.delete_gesture).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="✏️ Rename Gesture", command=self.rename_gesture).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🗂️ Delete Image", command=self.delete_image).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="❌ Quit", command=self.on_close).pack(side="left", padx=5)

    # ---------------- Checkbox ----------------
    def on_tree_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        col = self.tree.identify_column(event.x)
        if col != "#1":
            return
        path = self.tree.item(item_id)["values"][2]
        var = self.checkbox_vars[path]
        var.set(not var.get())
        self.tree.item(item_id, values=("✔" if var.get() else "⬜",
                                        self.tree.item(item_id)["values"][1],
                                        path))
        if var.get():
            self.tree.item(item_id, tags=('selected',))
            self.tree.tag_configure('selected', background='red')
        else:
            self.tree.item(item_id, tags=())

    def on_model_tree_click(self, event):
        item_id = self.model_tree.identify_row(event.y)
        if not item_id:
            return
        col = self.model_tree.identify_column(event.x)
        if col != "#1":
            return
        path = self.model_tree.item(item_id)["values"][1]
        var = self.model_checkbox_vars[path]
        var.set(not var.get())
        self.model_tree.item(item_id, values=("✔" if var.get() else "⬜", path))
        if var.get():
            self.model_tree.item(item_id, tags=('selected',))
            self.model_tree.tag_configure('selected', background='red')
        else:
            self.model_tree.item(item_id, tags=())

    # ---------------- Capture ----------------
    def toggle_capture(self):
        selected_paths = [p for p, var in self.checkbox_vars.items() if var.get()]
        if not selected_paths:
            messagebox.showwarning("No Selection", "Select at least one gesture to capture images.")
            return
        self.capturing = not self.capturing
        state = "ON" if self.capturing else "OFF"
        messagebox.showinfo("Capture", f"Capture {state} for selected gestures.")

    # ---------------- Gesture List ----------------
    def refresh_gesture_list(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.checkbox_vars.clear()
        for gesture in os.listdir(DATASET_PATH):
            gesture_dir = os.path.join(DATASET_PATH, gesture)
            if os.path.isdir(gesture_dir):
                count = len([f for f in os.listdir(gesture_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                var = tk.BooleanVar(value=False)
                self.checkbox_vars[gesture_dir] = var
                self.tree.insert("", "end", values=("⬜", count, gesture_dir))

    def add_gesture(self):
        gesture = self.entry_gesture.get().strip()
        if not gesture:
            messagebox.showwarning("No Name", "Enter a name for the new gesture.")
            return
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        if os.path.exists(gesture_dir):
            messagebox.showwarning("Already Exists", f"Gesture '{gesture}' already exists.")
            return
        os.makedirs(gesture_dir, exist_ok=True)
        self.refresh_gesture_list()
        self.entry_gesture.delete(0, tk.END)

    # ---------------- Delete / Rename / Image ----------------
    def delete_gesture(self):
        to_delete = [p for p, var in self.checkbox_vars.items() if var.get()]
        if not to_delete:
            messagebox.showwarning("Select Gesture", "Select gestures to delete.")
            return
        for path in to_delete:
            if messagebox.askyesno("Confirm", f"Delete gesture folder {path}?"):
                shutil.rmtree(path)
        self.refresh_gesture_list()

    def rename_gesture(self):
        to_rename = [p for p, var in self.checkbox_vars.items() if var.get()]
        if len(to_rename) != 1:
            messagebox.showwarning("Select One", "Select exactly one gesture to rename.")
            return
        path = to_rename[0]
        new_name = simpledialog.askstring("Rename Gesture", "Enter new name:")
        if new_name:
            new_path = os.path.join(DATASET_PATH, new_name)
            os.rename(path, new_path)
        self.refresh_gesture_list()

    def delete_image(self):
        to_select = [p for p, var in self.checkbox_vars.items() if var.get()]
        if len(to_select) != 1:
            messagebox.showwarning("Select One", "Select exactly one gesture to delete images from.")
            return
        path = to_select[0]
        files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            messagebox.showinfo("No Images", "No images in this gesture.")
            return
        file = filedialog.askopenfilename(initialdir=path, title="Select image to delete", filetypes=[("Image files","*.jpg;*.jpeg;*.png")])
        if file and os.path.exists(file):
            os.remove(file)
        self.refresh_gesture_list()

    # ---------------- Train ----------------
    def train_selected(self):
        selected_paths = [p for p, var in self.checkbox_vars.items() if var.get()]
        if not selected_paths:
            messagebox.showwarning("No Selection", "Select gestures to train.")
            return

        def _train():
            with tempfile.TemporaryDirectory() as tempdir:
                copied = False
                for gesture_dir in selected_paths:
                    images = [f for f in os.listdir(gesture_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if not images:
                        continue
                    gesture_name = os.path.basename(gesture_dir)
                    target_dir = os.path.join(tempdir, gesture_name)
                    os.makedirs(target_dir, exist_ok=True)
                    for f in images:
                        shutil.copy(os.path.join(gesture_dir, f), os.path.join(target_dir, f))
                    copied = True
                if not copied:
                    print("No valid gestures selected for training.")
                    return

                self.progress["value"] = 0
                self.root.update_idletasks()

                try:
                    data = gesture_recognizer.Dataset.from_folder(
                        dirname=tempdir,
                        hparams=gesture_recognizer.HandDataPreprocessingParams()
                    )
                    train_data, rest_data = data.split(0.8)
                    validation_data, test_data = rest_data.split(0.5)

                    export_path = os.path.join(os.getcwd(), f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    hparams = gesture_recognizer.HParams(export_dir=export_path)
                    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
                    model = gesture_recognizer.GestureRecognizer.create(
                        train_data=train_data,
                        validation_data=validation_data,
                        options=options
                    )
                    loss, acc = model.evaluate(test_data, batch_size=1)
                    model.export_model()
                    print(f"Training finished. Accuracy: {acc:.2f}, Loss: {loss:.2f}")

                    self.trained_models.append(export_path)
                    self.refresh_model_list()
                    self.progress["value"] = 100
                    self.root.update_idletasks()
                except Exception as e:
                    print("Training failed:", e)

        self.root.after(100, _train)

    # ---------------- Export ----------------
    def refresh_model_list(self):
        for i in self.model_tree.get_children():
            self.model_tree.delete(i)
        self.model_checkbox_vars.clear()
        for path in self.trained_models:
            var = tk.BooleanVar(value=False)
            self.model_checkbox_vars[path] = var
            self.model_tree.insert("", "end", values=("⬜", path))

    def export_selected_models(self):
        selected_paths = [p for p, var in self.model_checkbox_vars.items() if var.get()]
        if not selected_paths:
            messagebox.showwarning("No Selection", "Select trained models to export.")
            return
        dest = filedialog.askdirectory(title="Select Export Directory")
        if not dest:
            return
        for model_path in selected_paths:
            try:
                shutil.copytree(model_path, os.path.join(dest, os.path.basename(model_path)), dirs_exist_ok=True)
                print(f"Exported {model_path} to {dest}")
            except Exception as e:
                print("Export failed:", e)

    # ---------------- Close ----------------
    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
