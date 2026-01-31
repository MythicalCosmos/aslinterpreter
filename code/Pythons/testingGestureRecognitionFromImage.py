# gesture_gui_trainer_stable.py
import os
import cv2
import tkinter as tk
from tkinter import messagebox
from mediapipe_model_maker import gesture_recognizer
import mediapipe as mp

# ----------------------------
# Setup MediaPipe Hands
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Globals
# ----------------------------
DATA_DIR = "gesture_data"
current_gesture = None
current_hand = None
samples = []

# ----------------------------
# Functions
# ----------------------------
def capture_sample():
    global samples, current_gesture, current_hand
    if not current_gesture or not current_hand:
        messagebox.showerror("Error", "Set gesture name and hand first!")
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        messagebox.showwarning("Warning", "No hand detected!")
        return

    hand_landmarks = results.multi_hand_landmarks[0]
    flattened = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

    # Save CSV
    folder = os.path.join(DATA_DIR, current_gesture, current_hand)
    os.makedirs(folder, exist_ok=True)
    sample_path = os.path.join(folder, f"{len(os.listdir(folder))}.csv")
    with open(sample_path, "w") as f:
        f.write(",".join(map(str, flattened)))

    samples.append(sample_path)
    messagebox.showinfo("Captured", f"Sample saved: {sample_path}")

def set_gesture():
    global current_gesture
    current_gesture = gesture_entry.get().strip()
    if current_gesture:
        messagebox.showinfo("Gesture Set", f"Current gesture: {current_gesture}")

def set_hand(hand_side):
    global current_hand
    current_hand = hand_side
    messagebox.showinfo("Hand Set", f"Current hand: {current_hand}")

def train_model():
    if not os.path.exists(DATA_DIR):
        messagebox.showerror("Error", "No data to train!")
        return

    dataset = gesture_recognizer.Dataset.from_folder(DATA_DIR)
    train_data, val_data = dataset.split(0.8)

    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=val_data,
        epochs=30,
        batch_size=16
    )
    model.export_model("custom_gesture_recognizer.task")
    messagebox.showinfo("Training Complete", "Model exported as custom_gesture_recognizer.task")

# ----------------------------
# GUI Setup
# ----------------------------
root = tk.Tk()
root.title("Gesture Trainer")

gesture_entry = tk.Entry(root)
gesture_entry.pack(pady=5)
gesture_button = tk.Button(root, text="Set Gesture Name", command=set_gesture)
gesture_button.pack(pady=5)

hand_frame = tk.Frame(root)
hand_frame.pack(pady=5)
tk.Button(hand_frame, text="Left Hand", command=lambda: set_hand("left")).pack(side=tk.LEFT, padx=5)
tk.Button(hand_frame, text="Right Hand", command=lambda: set_hand("right")).pack(side=tk.LEFT, padx=5)

capture_button = tk.Button(root, text="Capture Sample", command=capture_sample)
capture_button.pack(pady=10)

train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.pack(pady=10)

# ----------------------------
# OpenCV Webcam Loop
# ----------------------------
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Webcam", frame)
    root.after(30, update_frame)

root.after(0, update_frame)
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), cv2.destroyAllWindows(), root.destroy()))
root.mainloop()
