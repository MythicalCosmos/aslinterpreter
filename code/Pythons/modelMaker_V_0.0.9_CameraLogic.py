#!/usr/bin/env python3
import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from datetime import datetime
#from mediapipe_model_maker import gesture_recognizer

class GestureDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Data Collector")

        self.dataset_path = os.path.join(os.getcwd(), "gesture_data")
        os.makedirs(self.dataset_path, exist_ok=True)

        self.gestures = self.load_existing_gestures()
        self.current_gesture = None
        self.capturing = False
        self.cap = None

        self.build_gui()
        self.init_camera()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        if self.cap:
            self.update_frame()

    # ---------------- Camera detection ----------------
    def init_camera(self):
        for i in range(4):  # Try the first 4 possible camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    print(f"✅ Camera found at index {i}")
                    return
            cap.release()

        messagebox.showerror("Camera Error", "No working webcam detected.")
        self.cap = None

    # ---------------- GUI setup ----------------
    def build_gui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        self.video_label = ttk.Label(frame)
        self.video_label.pack(pady=10)

        control_frame = ttk.Frame(frame)
        control_frame.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Current Gesture:").pack(side="left", padx=5)
        self.gesture_var = tk.StringVar()
        self.gesture_menu = ttk.Combobox(
            control_frame, textvariable=self.gesture_var, values=list(self.gestures.keys())
        )
        self.gesture_menu.pack(side="left", padx=5)

        ttk.Button(control_frame, text="Add Gesture", command=self.add_gesture).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Start Capture", command=self.toggle_capture).pack(side="left", padx=5)
        #ttk.Button(control_frame, text="Export Dataset", command=self.export_dataset).pack(side="left", padx=5)

        self.info_label = ttk.Label(frame, text="")
        self.info_label.pack(pady=5)

        self.refresh_gesture_info()

    # ---------------- Gesture management ----------------
    def load_existing_gestures(self):
        gestures = {}
        for name in os.listdir(self.dataset_path):
            gesture_dir = os.path.join(self.dataset_path, name)
            if os.path.isdir(gesture_dir):
                gestures[name] = len(os.listdir(gesture_dir))
        return gestures

    def add_gesture(self):
        name = simpledialog.askstring("New Gesture", "Enter gesture name:")
        if not name:
            return
        gesture_dir = os.path.join(self.dataset_path, name)
        os.makedirs(gesture_dir, exist_ok=True)
        self.gestures[name] = len(os.listdir(gesture_dir))
        self.gesture_menu["values"] = list(self.gestures.keys())
        self.gesture_var.set(name)
        self.refresh_gesture_info()

    def refresh_gesture_info(self):
        info = "\n".join(
            [f"{name}: {count} images" for name, count in self.gestures.items()]
        )
        self.info_label.config(text=info if info else "No gestures yet.")

    # ---------------- Camera capture ----------------
    def toggle_capture(self):
        if not self.cap:
            messagebox.showerror("Error", "No camera available.")
            return
        gesture = self.gesture_var.get()
        if not gesture:
            messagebox.showwarning("No Gesture Selected", "Please select or add a gesture first.")
            return

        self.current_gesture = gesture
        self.capturing = not self.capturing
        state = "ON" if self.capturing else "OFF"
        print("Capture", f"Capture {state} for '{gesture}' gesture")

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2.resize(rgb, (800, 600)))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                if self.capturing and self.current_gesture:
                    gesture_dir = os.path.join(self.dataset_path, self.current_gesture)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    filename = os.path.join(gesture_dir, f"{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    self.gestures[self.current_gesture] = len(os.listdir(gesture_dir))
                    self.refresh_gesture_info()
        self.root.after(30, self.update_frame)

    # ---------------- Export for MediaPipe ----------------

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureDataCollector(root)
    root.mainloop()
