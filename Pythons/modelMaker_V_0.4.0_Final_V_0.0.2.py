#!/usr/bin/env python3
import os
import cv2
import random
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Label,  Entry, StringVar, Frame, Scale, Listbox, HORIZONTAL, filedialog, messagebox, ttk, OptionMenu, IntVar
from PIL import Image, ImageTk
from datetime import datetime
from pathlib import Path
import re
import threading
import time
import shutil
import ttkbootstrap as tb
import sys
import matplotlib.pyplot as plt
#import tensorflow as tf
#assert tf.__version__.startswith('2')
#from mediapipe_model_maker import gesture_recognizer
DATASET_PATH = Path(r"~\SoftwareDev\SoftwareDev2\gesture_data")
EXPORT_PATH = Path(r"~\SoftwareDev\SoftwareDev2\exported_model")
DATASET_PATH.mkdir(parents=True, exist_ok=True)
EXPORT_PATH.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_BASENAME = "gesture_recognizer.task"
CAMERA_INDEX = 0
def get_next_version_dir(DATASET_PATH: Path) -> Path:
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    version_re = re.compile(r"version_(\d+)$")
    highest = 0
    for d in DATASET_PATH.iterdir():
        if d.is_dir() and version_re.match(d.name):
            num = int(version_re.match(d.name).group(1))
            highest = max(highest, num)
    next_num = highest + 1
    version_dir = DATASET_PATH / f"version_{next_num}"
    return version_dir
def get_next_version_dir_export(EXPORT_PATH: Path) -> Path:
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    version_re = re.compile(r"version_(\d+)$")
    highest = 0
    for d in EXPORT_PATH.iterdir():
        if d.is_dir() and version_re.match(d.name):
            num = int(version_re.match(d.name).group(1))
            highest = max(highest, num)
    next_num = highest + 1
    export_dir = EXPORT_PATH / f"version_{next_num}"
    return export_dir
class TextRedirector:
    def __init__(self, text_widget, mirror_to_terminal=True):
        self.text_widget = text_widget
        self.mirror_to_terminal = mirror_to_terminal
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self.lock = threading.Lock()
    def write(self, message):
        if not message.strip():
            return
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        full_message = timestamp + message.strip()
        self.text_widget.after(0, self._append_to_textbox, full_message)
        if self.mirror_to_terminal:
            with self.lock:
                self._stdout.write(full_message + "\n")
                self._stdout.flush()
    def flush(self):
        pass
    def _append_to_textbox(self, message):
        self.text_widget.config(state="normal")
        self.text_widget.insert("end", message + "\n")
        self.text_widget.see("end")
        self.text_widget.config(state="disabled")
    def hook(self):
        sys.stdout = self
        sys.stderr = self
    def unhook(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dataset Manager")
        self.root.geometry("1200x800")
        self.root.bind("<F11>", lambda e: self.root.attributes("-fullscreen", True))
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.resizable(True, True)
        self.version_dir = get_next_version_dir(DATASET_PATH)
        self.export_dir = get_next_version_dir_export(EXPORT_PATH)
        self.status_text = StringVar(value=f"Version folder: {self.version_dir}")
        self.dataset_path = DATASET_PATH
        self.export_path = EXPORT_PATH
        os.makedirs(self.version_dir, exist_ok=True)
        self.gestures = self.load_existing_gestures()
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open webcam.")
            raise RuntimeError("Could not open webcam")
        self.capturing = False
        self.current_gesture = None
        self.frame = None
        self.current_gesture = tk.StringVar()
        self.current_hand = tk.StringVar(value="left")
        self.status = tk.StringVar(value="Ready.")
        self.setup_style()
        self.build_gui()
        sys.stdout = TextRedirector(self.status_output)
        sys.stderr = TextRedirector(self.status_output)
        self.init_camera()
        if self.cap:
            self.start_camera_thread()
            self.update_frame()
        self.gestures = self.load_existing_gestures()
        self.current_gesture = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    def build_gui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=0) 
        self.root.rowconfigure(1, weight=1) 
        self.root.rowconfigure(2, weight=1)
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")
        for col in range(7):
            control_frame.columnconfigure(col, weight=1)
        self.capture_button_text = tk.StringVar(value="Start Capture")
        ttk.Button(control_frame, textvariable=self.capture_button_text, command=self.toggle_capture).grid(row=0, column=0, sticky="ew", padx=5)
        ttk.Button(control_frame, text="Refresh Gestures", command=self.refresh_gesture_info).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Visualize Model", command=self.visualize_data).grid(row=0,column=2, padx=5)
        ttk.Button(control_frame, text="Train & Export Model", command=self.train_data).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Delete Selected Label", command=self.delete_selected_label).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Open Version Folder", command=self.open_version_folder).grid(row=0, column=5, padx=5)
        ttk.Button(control_frame, text="Quit", command=self.on_close).grid(row=0, column=6, padx=5)
        gesture_frame = ttk.LabelFrame(self.root, text="Gesture Management")
        gesture_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        gesture_frame.columnconfigure(0, weight=1)
        gesture_frame.rowconfigure(1, weight=1)
        entry_frame = ttk.Frame(gesture_frame)
        entry_frame.grid(row=0, column=0, sticky="ew", pady=5)
        entry_frame.columnconfigure(1, weight=1)
        ttk.Label(entry_frame, text="New Gesture:").grid(row=0, column=0, sticky="w")
        self.new_label_entry = ttk.Entry(entry_frame)
        self.new_label_entry.grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(entry_frame, text="Add", command=self.add_gesture).grid(row=0, column=2, padx=5)
        table_frame = ttk.PanedWindow(gesture_frame, orient="horizontal")
        table_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        gesture_frame.rowconfigure(1, weight=1)
        gesture_frame.columnconfigure(0, weight=1)
        gesture_frame.rowconfigure(1, weight=1)
        gesture_frame.columnconfigure(0, weight=1)
        list_frame = ttk.Frame(table_frame)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.gesture_list = tk.Listbox(list_frame, exportselection=False, width=25)
        self.gesture_list.grid(row=0, column=0, sticky="nsew")
        self.gesture_list.bind("<<ListboxSelect>>", self.on_gesture_select)    
        scrollbar_list = ttk.Scrollbar(list_frame, orient="vertical", command=self.gesture_list.yview)
        scrollbar_list.grid(row=0, column=1, sticky="ns")
        self.gesture_list.config(yscrollcommand=scrollbar_list.set)    
        tree_frame = ttk.Frame(table_frame)
        tree_frame.grid(row=0, column=1, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tree_frame, columns=("version", "count"), show="headings")  
        self.tree.heading("version", text="Model Version")
        self.tree.heading("count", text="Image Count")  
        self.tree.column("version", width=150, anchor="center")
        self.tree.column("count", width=150, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        scrollbar_tree_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar_tree_y.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar_tree_y.set)
        table_frame.add(list_frame, weight=1)
        table_frame.add(tree_frame, weight=3)
        video_frame = ttk.LabelFrame(self.root, text="Live Camera Feed")
        video_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)  
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.status_output = tk.Text(status_frame, wrap="word", yscrollcommand=scrollbar.set, height=6, width=100, state="disabled")
        self.status_output.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.status_output.yview)
        version_frame = ttk.Label(self.root, text=f"{self.version_dir}")
        version_frame.grid(row=3, column=0, sticky="nsew", padx=10,pady=5)
        self.refresh_gesture_info()
    def setup_style(self):
        style = ttk.Style()
        self.root.configure(bg="#2f3136")
        try:
            style = tb.Style("darkly")
        except:
            pass
        dark_bg = "#2f3136"
        mid_bg = "#36393f"
        light_bg = "#40444b"
        accent = "#40444f"
        hover = "#40444f"
        text = "#dcddde"
        style.configure(".", background=dark_bg, foreground=text, font=("Segoe UI", 10))
        style.configure("TFrame", background=dark_bg)
        style.configure("TLabelframe", background=mid_bg, foreground=text, relief="flat")
        style.configure("TLabelframe.Label", background=mid_bg, foreground=text, font=("Segoe UI", 10, "bold"))
        style.configure("TButton", background=accent, foreground="white", borderwidth=0, focusthickness=3, focustcolor=hover, padding=6, relief="flat")
        style.map("TButton", background=[("active", hover), ("pressed", "#5F617A")], relief=[("pressed", "sunken"), ("!pressed", "flat")])
        style.configure("TLabel", background=dark_bg, foreground=text)
        style.configure("TEntry", fieldbackground=light_bg, foreground=text, borderwidth=1, relief="flat", insertcolor=text)
        style.map("TEntry", fieldbackground=[("active", mid_bg), ("!disabled", light_bg)])
        style.configure("TCombobox", fieldbackground=light_bg, background=light_bg, foreground=text)
        style.map("TCombobox", fieldbackground=[("readonly", light_bg)], selectbackground=[("readonly", light_bg)], selectforeground=[("readonly", text)])
        style.configure("Treeview", background=mid_bg, foreground=text, fieldbackground=mid_bg, borderwidth=0, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=accent, foreground="white", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", hover)], foreground=[("selected", "white")])
        self.root.option_add("*Listbox.background", mid_bg)
        self.root.option_add("*Listbox.foreground", text)
        self.root.option_add("*Listbox.selectBackground", hover)
        self.root.option_add("*Listbox.selectForeground", "white")
        self.root.option_add("*Listbox.font", ("Segoe UI", 10))
        if hasattr(self, "status_output"):
            self.status_output.configure(bg=mid_bg, fg=text, insertbackground=text, highlightbackground=dark_bg, relief="flat")
        style.configure("Vertical.TScrollbar", background=mid_bg, troughcolor=dark_bg, borderwidth=0, arrowcolor=text)
        style.configure("Horizontal.TScrollbar", background=mid_bg, troughcolor=dark_bg, borderwidth=0, arrowcolor=text)   
    def load_existing_gestures(self):
        gestures = {}
        for name in os.listdir(self.version_dir):
            gesture_dir = os.path.join(self.version_dir, name)
            if os.path.isdir(gesture_dir):
                gestures[name] = len(os.listdir(gesture_dir))
        return gestures
    def add_gesture(self):
        name = self.new_label_entry.get().strip()
        if not name:
            self.log_status("No name", "Please enter a gesture name.")
            return
        label = name.replace(" ", "_")
        entry_name = f"{label}"
        existing = [self.gesture_list.get(i) for i in range(self.gesture_list.size())]
        if entry_name in existing:
            self.log_status("Exists", "That label already exists.")
            return
        self.gesture_list.insert("end", entry_name)
        self.log_status(f"Added label {entry_name}")
        (self.version_dir / entry_name).mkdir(parents=True, exist_ok=True)
        self.new_label_entry.delete(0, "end")
        self.refresh_gesture_info()   
    def refresh_gesture_info(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.gestures = {}
        if not os.path.exists(self.version_dir):
            os.makedirs(self.version_dir, exist_ok=True)
        for gesture_name in os.listdir(self.version_dir):
            gesture_dir = os.path.join(self.version_dir, gesture_name)
            if os.path.isdir(gesture_dir):
                img_count = len([
                    f for f in os.listdir(gesture_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                self.gestures[gesture_name] = img_count
                self.tree.insert("", tk.END, values=(gesture_name, img_count))
    def delete_selected_label(self):
        sel = self.gesture_list.curselection()
        if not sel:
            return
        val = self.gesture_list.get(sel[0])
        if messagebox.askyesno("Confirm", f"Delete label entry '{val}'"):
            self.gesture_list.delete(sel[0])
            for item in self.tree.get_children():
                if self.tree.item(item, "text") == val:
                    self.tree.delete(item)
                    break
            self.current_gesture = val
            self.rm = os.path.join(self.version_dir, self.current_gesture)
            try:
                if os.path.isdir(self.rm):
                    shutil.rmtree(self.rm)  # removes directories
                elif os.path.isfile(self.rm):
                    os.remove(self.rm)      # removes files
                self.log_status(f"Removed {val}")
            except PermissionError:
                messagebox.showerror("Error", f"Cannot delete '{self.rm}'. It may be open or locked.")
            self.log_status(f"Removed {val}")
        self.refresh_gesture_info()
    def open_version_folder(self):
        os.startfile(self.version_dir)
    def get_selected_label(self):
        sel = self.gesture_list.get()
        if not gesture:
            sel = self.gesture_list.curselection()
            if sel:
                gesture = self.gesture_list.get(sel[0])
    def on_gesture_select(self, event):
        selection = self.gesture_list.curselection()
        if selection:
            selected_gesture = self.gesture_list.get(selection[0])
            self.log_status(f"Gesture selected: '{selected_gesture}'")
    def init_camera(self):
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            cap = self.cap
        else:
            cap = None
            for i in range(4):
                tmp = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if tmp.isOpened():
                    ret, _ = tmp.read()
                    if ret:
                        cap = tmp
                        break
                    tmp.release()
        if not cap or not cap.isOpened():
            messagebox.showerror("No camera found")
            self.cap = None
            return             
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        print(f"Found Camera {self.cap.isOpened()}") 
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.frame = None
        self.camera_thread = None             
    def camera_loop(self):
        while not getattr(self, "stop_event", threading.Event()).is_set() and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            with self.frame_lock:
                self.frame = frame    
            time.sleep(0.01)
    def start_camera_thread(self):
        if getattr(self, "camera_thread", None) and self.camera_thread.is_alive():
            return
        self.stop_event.clear()
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        print("Camera started")        
    def update_frame(self):
        if isinstance(self.current_gesture, tuple):
            self.current_gesture = str(self.current_gesture[0])
        frame_copy = None
        with self.frame_lock:
            if self.frame is not None:                    
                frame_copy = self.frame.copy()
        if frame_copy is not None:
            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2.resize(rgb, (550, 309)))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            if self.capturing and self.current_gesture:
                gesture_dir = os.path.join(self.version_dir, self.current_gesture)
                os.makedirs(gesture_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                filename = os.path.join(gesture_dir, f"{timestamp}.jpg")
                cv2.imwrite(filename, frame_copy)
                image_count = len([
                    f for f in os.listdir(gesture_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                self.gestures[self.current_gesture] = image_count
                if image_count % 500 == 0:
                    self.log_status(f"Captured {image_count} images for '{self.current_gesture}'")
        self.refresh_gesture_info()
        self.root.after(16, self.update_frame)
    def toggle_capture(self):
        if not self.cap:
            self.log_status("Error", "No camera available.")
            return
        sel = self.gesture_list.curselection()
        if sel:
            gesture = self.gesture_list.get(sel[0])
        else:
            self.log_status("No Gesture Selected", "Please select or add a gesture first.")
            return
        gesture = self.gesture_list.get(sel[0])
        self.current_gesture = gesture
        self.capturing = not self.capturing
        if self.capturing:
            self.capture_button_text.set("Stop Capture")
            gesture_dir = os.path.join(self.version_dir, gesture)
            os.makedirs(gesture_dir, exist_ok=True)
            existing_count = len(os.listdir(gesture_dir))
            self.log_status(f"Started Capture for gesture '{gesture}'.")
        else:
            self.capture_button_text.set("Start Capture")
            current_count = self.gestures.get(gesture, 0)
            self.log_status(f"Stopped capture for '{gesture}' (total: {current_count} images)")
        state = "ON" if self.capturing else "OFF"
        self.log_status(f"Capture {state} for '{gesture}' gesture")       
    def visualize_data(self, NUM_EXAMPLES=5):
            DATASET_PATH = self.version_dir
            print(DATASET_PATH)
            labels = []
            for i in os.listdir(DATASET_PATH):
                if os.path.isdir(os.path.join(DATASET_PATH, i)):
                    labels.append(i)
            print(labels)
            for label in labels:
                label_dir = os.path.join(DATASET_PATH, label)
                example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
                fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
                for i in range(NUM_EXAMPLES):
                    axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
                    axs[i].get_xaxis().set_visible(False)
                    axs[i].get_yaxis().set_visible(False)
                fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')
            plt.show()
            return labels
    def train_data(self):
        DATASET_PATH = self.version_dir
        data = gesture_recognizer.Dataset.from_folder(dirname=DATASET_PATH, hparams=gesture_recognizer.HandDataPreprocessingParams())
        train_data, rest_data = data.split(0.8)
        validation_data, test_data = rest_data.split(0.5)
        hparams = gesture_recognizer.HParams(export_dir=self.export_dir / "Exported")
        options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
        model = gesture_recognizer.GestureRecognizer.create(train_data=train_data, validation_data=validation_data, options=options)
        loss, acc = model.evaluate(test_data, batch_size=1)
        self.log_status(f"Test loss:{loss}, Test accuracy:{acc}")
        model.export_model()
        print("Exporting Model")
        hparams = gesture_recognizer.HParams(learning_rate=0.003, export_dir=self.export_dir / "Final Export")
        model_options = gesture_recognizer.ModelOptions(dropout_rate=0.2)
        options = gesture_recognizer.GestureRecognizerOptions(model_options=model_options, hparams=hparams)
        model_2 = gesture_recognizer.GestureRecognizer.create(train_data=train_data, validation_data=validation_data, options=options)
        loss, accuracy = model_2.evaluate(test_data)
        self.log_status(f"Test loss:{loss}, Test accuracy:{accuracy}")
        print("Model Exported Successfully")
    def log_status(self, message):
        print(message) 
    def on_close(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if getattr(self, "camera_thread", None):
            self.camera_thread.join(timeout=0.5)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.root.destroy()
if __name__ == "__main__":
    print(EXPORT_PATH)
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()