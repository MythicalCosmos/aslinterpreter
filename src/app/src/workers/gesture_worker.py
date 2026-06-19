import time
from typing import List, Optional
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import PyQt6.QtCore as qtc

from core.constants import LANDMARK_MODEL_PATH, CLASSIFIER_MODEL_PATH, LABELS_PATH


class GestureWorker(qtc.QObject):
    gestureRecognized = qtc.pyqtSignal(str, float)   # (label, confidence)
    landmarksReady    = qtc.pyqtSignal(object, tuple) # (landmarks, frame_shape)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.enabled               = False
        self.clf                   = None
        self.labels: List[str]     = []
        self.mp_hands              = None
        self.hand_tracking_enabled = False
        self.lastProcessTime       = 0.0
        self.minInterval           = 0.10   # max 10 inferences/sec
        self.lastEmitTime          = 0.0
        self.minEmitInterval       = 0.30   # max 3 emissions/sec

        self._load_classifier()
        self._load_labels()
        self._load_landmark_model()

    # ── Initialization ────────────────────────────────────────────────────

    def _load_classifier(self) -> None:
        try:
            self.clf = joblib.load(str(CLASSIFIER_MODEL_PATH))
            self.enabled = True
            print(f"[GestureWorker] Classifier loaded: {CLASSIFIER_MODEL_PATH.name}")
        except FileNotFoundError:
            print(f"[GestureWorker] No classifier at {CLASSIFIER_MODEL_PATH} — train first.")

    def _load_labels(self) -> None:
        try:
            self.labels = LABELS_PATH.read_text().splitlines()
            print(f"[GestureWorker] Labels: {self.labels}")
        except FileNotFoundError:
            print(f"[GestureWorker] No labels file at {LABELS_PATH}")

    def _load_landmark_model(self) -> None:
        if not LANDMARK_MODEL_PATH.exists():
            print(f"[GestureWorker] hand_landmarker.task not found — run scripts/setup.py")
            return
        try:
            base_options = python.BaseOptions(model_asset_path=str(LANDMARK_MODEL_PATH))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE,
            )
            self.mp_hands = vision.HandLandmarker.create_from_options(options)
            self.hand_tracking_enabled = True
            print("[GestureWorker] HandLandmarker initialized.")
        except Exception as e:
            print(f"[GestureWorker] HandLandmarker init failed: {e}")

    # ── Frame processing ──────────────────────────────────────────────────

    @qtc.pyqtSlot(np.ndarray)
    def processFrame(self, frame: np.ndarray) -> None:
        """
        Run hand landmark detection and gesture classification on one frame.
        Rate-limited to minInterval. Emits gestureRecognized and landmarksReady.
        """
        if not self.enabled or not self.hand_tracking_enabled:
            return
        now = time.time()
        if now - self.lastProcessTime < self.minInterval:
            return
        if frame is None:
            return
        self.lastProcessTime = now

        frame_flipped = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.mp_hands.detect(mp_image)

        if not result.hand_landmarks:
            return

        lm = result.hand_landmarks[0]
        wrist = lm[0]
        vec = np.array(
            [[l.x - wrist.x, l.y - wrist.y, l.z - wrist.z] for l in lm],
            dtype=np.float32
        ).flatten().reshape(1, 63)

        proba = self.clf.predict_proba(vec)[0]
        idx   = int(np.argmax(proba))
        score = float(proba[idx])
        label = self.labels[idx] if idx < len(self.labels) else str(idx)

        # Always emit landmarks for overlay
        self.landmarksReady.emit(result.hand_landmarks[0], frame_flipped.shape[:2])

        if score > 0.5 and now - self.lastEmitTime > self.minEmitInterval:
            self.lastEmitTime = now
            self.gestureRecognized.emit(label, score)

    def reload(self) -> None:
        """Reload classifier and labels from disk after new training."""
        self._load_classifier()
        self._load_labels()