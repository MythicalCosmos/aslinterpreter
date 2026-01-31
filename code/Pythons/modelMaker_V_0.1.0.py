#!/usr/bin/env python3
import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from datetime import datetime
#from mediapipe_model_maker import gesture_recognizer
import shutil

DATASET_PATH = os.path.expanduser("~/gesture_data")
os.makedirs(DATASET_PATH, exist_ok=True)

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dataset Manager")
        self.root.geometry("1000x750")
        self.root.resizable(False, False)

        self.cap = None
        self.detect_camera()

        self.current_gesture = tk.StringVar()
        self.current_hand = tk.StringVar(value="left")
        self.capturing = False

        self.build_gui()
        self.refresh_gesture_list()

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
        messagebox.showerror("Camera Error", "No working webcam detected.")
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

                if self.capturing and self.current_gesture.get():
                    gesture = self.current_gesture.get()
                    hand = self.current_hand.get()
                    gesture_dir = os.path.join(DATASET_PATH, hand, gesture)
                    os.makedirs(gesture_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filepath = os.path.join(gesture_dir, f"{timestamp}.jpg")
                    cv2.imwrite(filepath, frame)
                    self.refresh_gesture_list()
        self.root.after(30, self.update_frame)

    # ---------------- GUI ----------------
    def build_gui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=5)

        # Video preview
        self.video_label = ttk.Label(top_frame)
        self.video_label.pack()

        control_frame = ttk.Frame(top_frame)
        control_frame.pack(pady=5)

        ttk.Label(control_frame, text="Gesture:").grid(row=0, column=0, padx=5)
        self.entry_gesture = ttk.Entry(control_frame, textvariable=self.current_gesture, width=15)
        self.entry_gesture.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Hand:").grid(row=0, column=2, padx=5)
        self.combo_hand = ttk.Combobox(control_frame, textvariable=self.current_hand, values=["left", "right"], width=10)
        self.combo_hand.grid(row=0, column=3, padx=5)
        self.combo_hand.current(0)

        ttk.Button(control_frame, text="📸 Capture", command=self.toggle_capture).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Add Gesture", command=self.add_gesture).grid(row=0, column=5, padx=5)

        # Gesture table
        table_frame = ttk.LabelFrame(self.root, text="Gestures")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.tree = ttk.Treeview(table_frame, columns=("hand", "count", "path"), show="headings")
        self.tree.heading("hand", text="Hand")
        self.tree.heading("count", text="Image Count")
        self.tree.heading("path", text="Path")
        self.tree.column("hand", width=100)
        self.tree.column("count", width=100)
        self.tree.column("path", width=600)
        self.tree.pack(fill="both", expand=True)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="🔄 Refresh", command=self.refresh_gesture_list).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🧠 Train Selected", command=self.train_selected).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="❌ Quit", command=self.on_close).pack(side="left", padx=5)

    # ---------------- Capture ----------------
    def toggle_capture(self):
        if not self.current_gesture.get():
            messagebox.showwarning("No Gesture", "Enter a gesture name to capture images.")
            return
        self.capturing = not self.capturing
        state = "ON" if self.capturing else "OFF"
        messagebox.showinfo("Capture", f"Capture {state} for '{self.current_gesture.get()}'")

    # ---------------- Gesture Table ----------------
    def refresh_gesture_list(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for hand in ["left", "right"]:
            hand_dir = os.path.join(DATASET_PATH, hand)
            if os.path.exists(hand_dir):
                for gesture in os.listdir(hand_dir):
                    gesture_dir = os.path.join(hand_dir, gesture)
                    if os.path.isdir(gesture_dir):
                        count = len([f for f in os.listdir(gesture_dir) if f.lower().endswith(".jpg")])
                        self.tree.insert("", "end", values=(hand, count, gesture_dir))

    def add_gesture(self):
        gesture = simpledialog.askstring("Add Gesture", "Enter gesture name:")
        if not gesture:
            return
        hand = self.current_hand.get()
        gesture_dir = os.path.join(DATASET_PATH, hand, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        self.current_gesture.set(gesture)
        self.refresh_gesture_list()

    # ---------------- Train ----------------
    def train_selected(self):
        print("button works")
        '''
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Select gestures to train on.")
            return

        temp_dir = os.path.join(DATASET_PATH, "_selected")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        for item in selection:
            path = self.tree.item(item)["values"][2]
            hand = self.tree.item(item)["values"][0]
            gesture = os.path.basename(path)
            target_dir = os.path.join(temp_dir, hand, gesture)
            os.makedirs(target_dir, exist_ok=True)
            for f in os.listdir(path):
                if f.endswith(".jpg"):
                    shutil.copy(os.path.join(path, f), os.path.join(target_dir, f))

        # Train
        try:
            data = gesture_recognizer.Dataset.from_folder(
                dirname=temp_dir,
                hparams=gesture_recognizer.HandDataPreprocessingParams()
            )
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
            model.export_model()
            messagebox.showinfo("Training Complete", f"Accuracy: {acc:.2f}\nLoss: {loss:.2f}")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
'''
    # ---------------- Close ----------------
    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
