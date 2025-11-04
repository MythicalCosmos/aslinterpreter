#!/usr/bin/env python3
import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
#import tensorflow as tf
#from mediapipe_model_maker import gesture_recognizer
import matplotlib.pyplot as plt
import random

# === GLOBAL CONFIG ===
DATASET_PATH = os.path.expanduser("~/model_data")

# === CREATE BASE FOLDER STRUCTURE ===
os.makedirs(DATASET_PATH, exist_ok=True)

# === MAIN APP ===
class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dataset Maker")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.running = True

        # Current gesture and hand
        self.current_gesture = tk.StringVar()
        self.current_hand = tk.StringVar()

        # Build GUI
        self.build_gui()

        # Start webcam update loop
        self.update_frame()

    def build_gui(self):
        frame_controls = ttk.Frame(self.root)
        frame_controls.pack(pady=10)

        ttk.Label(frame_controls, text="Gesture name:").grid(row=0, column=0, padx=5)
        self.entry_gesture = ttk.Entry(frame_controls, textvariable=self.current_gesture, width=15)
        self.entry_gesture.grid(row=0, column=1, padx=5)

        ttk.Label(frame_controls, text="Hand:").grid(row=0, column=2, padx=5)
        self.combo_hand = ttk.Combobox(frame_controls, textvariable=self.current_hand,
                                       values=["left", "right"], width=10)
        self.combo_hand.grid(row=0, column=3, padx=5)
        self.combo_hand.current(0)

        self.btn_capture = ttk.Button(frame_controls, text="📸 Capture Image", command=self.capture_image)
        self.btn_capture.grid(row=0, column=4, padx=10)

        self.btn_train = ttk.Button(frame_controls, text="🧠 Train Model", command=self.train_model)
        self.btn_train.grid(row=0, column=5, padx=10)

        self.btn_quit = ttk.Button(frame_controls, text="❌ Quit", command=self.quit_app)
        self.btn_quit.grid(row=0, column=6, padx=10)

        ttk.Label(frame_controls, text="Max Images (0 = all):").grid(row=1, column=0, padx=5, pady=5)
        self.limit_var = tk.IntVar(value=0)
        self.limit_spin = ttk.Spinbox(frame_controls, from_=0, to=10000, textvariable=self.limit_var, width=10)
        self.limit_spin.grid(row=1, column=1, padx=5)


        # Video frame
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)

        # Status
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status, font=("Arial", 11)).pack(pady=5)

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

    def capture_image(self):
        print("button works")
        '''
        gesture = self.current_gesture.get().strip().lower()
        hand = self.current_hand.get().strip().lower()

        if not gesture or hand not in ["left", "right"]:
            messagebox.showwarning("Input Error", "Please enter a gesture name and select a hand.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Could not capture image.")
            return

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Create directories
        gesture_dir = os.path.join(DATASET_PATH, hand, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        filepath = os.path.join(gesture_dir, filename)
        cv2.imwrite(filepath, frame)

        self.status.set(f"Saved: {filepath}")

    def train_model(self):
        self.status.set("Training model, please wait...")
        self.root.update()

        try:
            limit = self.limit_var.get()
            data = gesture_recognizer.Dataset.from_folder(
                dirname=DATASET_PATH,
                hparams=gesture_recognizer.HandDataPreprocessingParams()
            )
            if limit > 0:
                # Shuffle and trim the dataset
                all_examples = list(data._dataset)  # internal dataset
                random.shuffle(all_examples)
                data._dataset = all_examples[:limit]
                self.status.set(f"Training on {limit} images (subset)")
            else:
                self.status.set(f"Training on all {len(data._dataset)} images")
            train_data, rest_data = data.split(0.8)
            validation_data, test_data = rest_data.split(0.5)

            hparams = gesture_recognizer.HParams(export_dir="exported_model_gui")
            options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
            model = gesture_recognizer.GestureRecognizer.create(
                train_data=train_data,
                validation_data=validation_data,
                options=options
            )

            loss, acc = model.evaluate(test_data, batch_size=1)
            messagebox.showinfo("Training Complete", f"Accuracy: {acc:.2f}\nLoss: {loss:.2f}")

            model.export_model()
            self.status.set("Model exported successfully to ./exported_model_gui")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.set("Training failed.")
            '''

    def quit_app(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

# === RUN ===
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
