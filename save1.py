import cv2
import tkinter as tk
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# ---------------------------
# Tkinter setup
# ---------------------------
root = tk.Tk()
root.title("Recognized Gestures")
root.geometry("400x150")

gesture_label = tk.Label(root, text="Waiting for gestures...", font=("Helvetica", 20))
gesture_label.pack(pady=40)

def update_label(text):
    gesture_label.config(text=text)
    root.update_idletasks()  # Refresh GUI

# ---------------------------
# MediaPipe Gesture Recognizer
# ---------------------------
model_path = '/Users/dawsonrieple/Desktop/SoftwareDev/gesture_recognizer.task'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ---------------------------
# Frame capture + gesture update
# ---------------------------
def update_gestures():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        result = recognizer.recognize(mp_image)

        if result.gestures:
            gesture_texts = []
            for hand_gestures in result.gestures:
                if hand_gestures:
                    top_gesture = hand_gestures[0]
                    gesture_texts.append(f"{top_gesture.category_name} ({top_gesture.score:.2f})")
            update_label(" | ".join(gesture_texts))
        else:
            update_label("No gestures detected")

    root.after(50, update_gestures)  # Repeat every 50 ms

# Start the loop
update_gestures()

# ---------------------------
# Clean exit
# ---------------------------
def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# Start Tkinter mainloop
root.mainloop()
