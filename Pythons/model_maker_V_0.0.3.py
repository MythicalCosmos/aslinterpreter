#!/usr/bin/env python3
import os
import cv2
import random
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime
#import tensorflow as tf
#from mediapipe_model_maker import gesture_recognizer

DATASET_PATH = os.path.expanduser("~/gesture_data")
os.makedirs(DATASET_PATH, exist_ok=True)


class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dataset Manager")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.current_gesture = tk.StringVar()
        self.current_hand = tk.StringVar(value="left")
        self.status = tk.StringVar(value="Ready.")

        self.build_gui()
        self.update_frame()
        self.refresh_gesture_list()

    # === GUI Layout ===
    def build_gui(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        ttk.Label(control_frame, text="Gesture:").grid(row=0, column=0, padx=5)
        self.entry_gesture = ttk.Entry(control_frame, textvariable=self.current_gesture, width=15)
        self.entry_gesture.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Hand:").grid(row=0, column=2, padx=5)
        self.combo_hand = ttk.Combobox(control_frame, textvariable=self.current_hand, values=["left", "right"], width=10)
        self.combo_hand.grid(row=0, column=3, padx=5)

        ttk.Button(control_frame, text="📸 Capture", command=self.capture_image).grid(row=0, column=4, padx=10)
        ttk.Button(control_frame, text="🔄 Refresh Gestures", command=self.refresh_gesture_list).grid(row=0, column=5, padx=10)
        ttk.Button(control_frame, text="🧠 Train Selected", command=self.train_selected).grid(row=0, column=6, padx=10)
        ttk.Button(control_frame, text="❌ Quit", command=self.quit_app).grid(row=0, column=7, padx=10)

        # Video preview
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)

        # Gesture list
        list_frame = ttk.LabelFrame(self.root, text="📁 Gestures in Dataset")
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.tree = ttk.Treeview(list_frame, columns=("hand", "count", "path"), show="headings")
        self.tree.heading("hand", text="Hand")
        self.tree.heading("count", text="Image Count")
        self.tree.heading("path", text="Path")
        self.tree.column("hand", width=100)
        self.tree.column("count", width=100)
        self.tree.column("path", width=600)
        self.tree.pack(fill="both", expand=True)

        # Status
        ttk.Label(self.root, textvariable=self.status, font=("Arial", 11)).pack(pady=5)

    # === Webcam ===
    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    # === Capture ===
    def capture_image(self):
        gesture = self.current_gesture.get().strip().lower()
        hand = self.current_hand.get().strip().lower()

        if not gesture or hand not in ["left", "right"]:
            messagebox.showwarning("Input Error", "Enter a gesture and select a hand.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Could not capture image.")
            return

        frame = cv2.flip(frame, 1)
        gesture_dir = os.path.join(DATASET_PATH, hand, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        filepath = os.path.join(gesture_dir, filename)
        cv2.imwrite(filepath, frame)

        self.status.set(f"Saved: {filepath}")
        self.refresh_gesture_list()

    # === Refresh Gesture Table ===
    def refresh_gesture_list(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

        for hand in ["left", "right"]:
            hand_dir = os.path.join(DATASET_PATH, hand)
            if not os.path.exists(hand_dir):
                continue
            for gesture in sorted(os.listdir(hand_dir)):
                gpath = os.path.join(hand_dir, gesture)
                if os.path.isdir(gpath):
                    count = len([f for f in os.listdir(gpath) if f.lower().endswith(".jpg")])
                    self.tree.insert("", "end", values=(hand, count, gpath))
        self.status.set("Gesture list refreshed.")

    # === Train on selected gestures ===
    def train_selected(self):
        print("button works")
        '''
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Select at least one gesture to train.")
            return

        selected_dirs = [self.tree.item(i)["values"][2] for i in selection]
        temp_dataset = os.path.join(DATASET_PATH, "_selected")
        os.makedirs(temp_dataset, exist_ok=True)

        # Build temp folder for selected gestures
        import shutil
        shutil.rmtree(temp_dataset)
        os.makedirs(temp_dataset, exist_ok=True)
        for gdir in selected_dirs:
            hand = os.path.basename(os.path.dirname(gdir))
            gesture = os.path.basename(gdir)
            target_dir = os.path.join(temp_dataset, hand, gesture)
            os.makedirs(target_dir, exist_ok=True)
            for file in os.listdir(gdir):
                if file.endswith(".jpg"):
                    src = os.path.join(gdir, file)
                    dst = os.path.join(target_dir, file)
                    shutil.copy(src, dst)

        # Train model
        try:
            self.status.set("Training on selected gestures...")
            self.root.update()

            data = gesture_recognizer.Dataset.from_folder(
                dirname=temp_dataset,
                hparams=gesture_recognizer.HandDataPreprocessingParams()
            )
            train_data, rest_data = data.split(0.8)
            validation_data, test_data = rest_data.split(0.5)

            hparams = gesture_recognizer.HParams(export_dir="exported_model_gui_v3")
            options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
            model = gesture_recognizer.GestureRecognizer.create(
                train_data=train_data,
                validation_data=validation_data,
                options=options
            )

            loss, acc = model.evaluate(test_data, batch_size=1)
            messagebox.showinfo("Training Complete", f"Accuracy: {acc:.2f}\nLoss: {loss:.2f}")
            model.export_model()
            self.status.set("Model exported successfully to ./exported_model_gui_v3")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self.status.set("Training failed.")
            '''

    # === Quit ===
    def quit_app(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()