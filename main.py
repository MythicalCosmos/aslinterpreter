#!/usr/bin/env python3
import time
_BOOT_T0 = time.perf_counter()
from config.loader import loadSettings
from config.loadDefaults import loadDefaultSettings
from config.writer import ConfigAPI

# ── Standard library ──────────────────────────────────────────────────────
import os
import io
import json
import shutil
import sys
import threading
import time
import platform
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional, List
from difflib import get_close_matches
import queue

# ── Third-party ───────────────────────────────────────────────────────────
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import joblib
import soundcard as sc
import soundfile as sf
from faster_whisper import WhisperModel

# ── Qt ────────────────────────────────────────────────────────────────────
import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg

# ── Internal ──────────────────────────────────────────────────────────────
from core.loadLogging import setupLogging
from core.crashlogger import installCrashHandler

installCrashHandler()

def boot_log(msg):
    print(f"[BOOT +{time.perf_counter() - _BOOT_T0:6.2f}s] {msg}", flush=True)

# ── Settings & constants (loaded once) ────────────────────────────────────
SETTINGS               = loadSettings()
HF_TOKEN               = SETTINGS.env.hf_token
VERSION                = SETTINGS.version.version
CAMERA_INDEX           = SETTINGS.app.camera
NUM_EXAMPLES           = SETTINGS.settings.examples
SAMPLE_RATE            = SETTINGS.settings.sam_rate
INITIAL_CHUNK_DURATION = SETTINGS.settings.init_chunk_der
MIN_CHUNK_DURATION     = SETTINGS.settings.min_chunk_der
CHUNK_DECREMENT        = SETTINGS.settings.chunk_dec
CONFIDENCE_THRESHOLD   = SETTINGS.settings.confidence_threshold
WORD_GAP               = SETTINGS.settings.word_gap
AUTOCORRECT_TOGGLE     = SETTINGS.settings.autocorrect
AUTOCORRECT_THRESHOLD  = SETTINGS.settings.autocorrect_threshold
LOG_LEVEL              = SETTINGS.app.log_level
MODEL_NAME             = SETTINGS.gestures.gesture_model
IMAGE_EXTENSIONS       = (".png", ".jpg", ".jpeg", ".webp")

# ── Path constants ────────────────────────────────────────────────────────
_SRC_ROOT  = Path(__file__).resolve().parent       # src/app/src/
_REPO_ROOT = _SRC_ROOT.parent.parent.parent        # src/

LOG_DIR               = _REPO_ROOT / "shared/logs"
SHARED                = _REPO_ROOT / "shared"
DATASET_PATH          = _REPO_ROOT / "shared/datasets"
EXPORT_PATH           = _REPO_ROOT / "shared/exports"
LANDMARK_MODEL_PATH   = _REPO_ROOT / "src/deploy/hand_landmarker.task"
CLASSIFIER_MODEL_PATH = _REPO_ROOT / "src/deploy/asl_model.pkl"
MODEL_PATH            = CLASSIFIER_MODEL_PATH   # legacy alias
LABELS_PATH           = _REPO_ROOT / "src/deploy/labels.txt"
WORDLIST              = _REPO_ROOT / "src/deploy/words.txt"
JSON_FILE             = _REPO_ROOT / "shared/gestures.json"
WORKER_LOG_PATH       = _REPO_ROOT / "shared/logs/training_log.txt"

APP_LOG = setupLogging(LOG_DIR)

HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# ── Helpers ───────────────────────────────────────────────────────────────

def findWorkingCamera(start_index: int = 0, max_tested: int = 5) -> Optional[int]:
    """Return first camera index that opens successfully, else None."""
    for i in range(start_index, max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None


# ── Enums / simple classes ────────────────────────────────────────────────

class LogLevel(IntEnum):
    DEBUG   = 0
    INFO    = 1
    WARNING = 2
    ERROR   = 3


class WindowMode:
    WINDOWED   = "windowed"
    FULLSCREEN = "fullscreen"
    BORDERLESS = "borderless"


# ── WindowManager ─────────────────────────────────────────────────────────

class WindowManager:
    def __init__(self, window, settings) -> None:
        self.window       = window
        self.settings     = settings
        self.mode         = settings.app.fullscreen_mode
        self.width        = settings.app.width
        self.height       = settings.app.height
        self.monitorIndex = settings.app.monitor
        self.posx         = settings.app.pos_x
        self.posy         = settings.app.pos_y
        self.dpiScaling   = settings.app.dpi_scaling

    def screens(self):
        return qtw.QApplication.screens()

    def currentScreen(self):
        screens = self.screens()
        return screens[min(self.monitorIndex, len(screens) - 1)]

    def availableResolutions(self):
        screen = self.currentScreen()
        size   = screen.size()
        w, h   = size.width(), size.height()
        common = [
            (3840,2160),(2560,1440),(2048,1536),(1920,1440),(1920,1080),
            (1600,900),(1400,1050),(1280,720),(1280,960),
            (1024,768),(800,600),(640,480),
        ]
        return sorted([(cw, ch) for cw, ch in common if cw <= w and ch <= h], reverse=True)

    def apply(self, mode=None, width=None, height=None, monitor=None) -> None:
        if mode    is not None: self.mode         = mode
        if width:               self.width        = width
        if height:              self.height       = height
        if monitor is not None: self.monitorIndex = monitor

        w      = self.window
        screen = self.currentScreen()
        w.showNormal()
        w.setWindowFlags(qtc.Qt.WindowType.Window)
        w.move(self.posx, self.posy)

        if self.mode == WindowMode.WINDOWED:
            w.resize(self.width, self.height)
            w.move(self.posx, self.posy)
            w.show()
        elif self.mode == WindowMode.FULLSCREEN:
            w.windowHandle().setScreen(screen)
            w.showFullScreen()
        elif self.mode == WindowMode.BORDERLESS:
            w.setWindowFlags(qtc.Qt.WindowType.FramelessWindowHint)
            w.windowHandle().setScreen(screen)
            w.setGeometry(screen.geometry())
            w.show()

    def saveState(self) -> None:
        g         = self.window.geometry()
        self.posx = g.x()
        self.posy = g.y()
        self.width  = g.width()
        self.height = g.height()

    def applyDPI(self) -> None:
        if self.dpiScaling:
            qtw.QApplication.setHighDpiScaleFactorRoundingPolicy(
                qtc.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )


# ── UILogger ──────────────────────────────────────────────────────────────

class UILogger(qtc.QObject):
    logReady = qtc.pyqtSignal(str)

    def __init__(self, name: str = "app", level: int = LogLevel.INFO) -> None:
        super().__init__()
        self.name  = name
        self.level = level

    def setLevel(self, level: int) -> None:
        self.level = level

    def log(self, message, level: int) -> None:
        if level < self.level:
            return
        labels = {LogLevel.DEBUG: "Debug", LogLevel.INFO: "Info",
                  LogLevel.WARNING: "Warning", LogLevel.ERROR: "Error"}
        ts = datetime.now().strftime("[%H:%M:%S]")
        self.logReady.emit(f"{labels.get(level, 'Info')} {ts} {message}")


# ── LogViewer ─────────────────────────────────────────────────────────────

class LogViewer(qtw.QTextEdit):
    def __init__(self, maxLines: int = 1500, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.maxLines  = maxLines
        self.lines: List[str]    = []
        self.pending: List[str]  = []
        self.flushTimer = qtc.QTimer(self)
        self.flushTimer.timeout.connect(self.flush)
        self.flushTimer.start(100)

    def enqueue(self, message: str) -> None:
        self.pending.append(message)

    def flush(self) -> None:
        if not self.pending:
            return
        self.lines.extend(self.pending)
        self.pending.clear()
        if len(self.lines) > self.maxLines:
            self.lines = self.lines[-self.maxLines:]
        self.setPlainText("\n".join(self.lines))
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ── WhisperManager ────────────────────────────────────────────────────────

class WhisperManager:
    _model = None

    @classmethod
    def getModel(cls):
        if cls._model is None:
            print("Loading Whisper model...")
            boot_log("loading Whisper model...")
            cls._model = WhisperModel("small", device="cpu", compute_type="int8")
            boot_log("Whisper model loaded")
            print("Model loaded.")
        return cls._model


# ── WordDecoder ───────────────────────────────────────────────────────────

class WordDecoder:
    def __init__(self, wordSet) -> None:
        self.wordSet  = wordSet
        self.buffer: List[str]       = []
        self.lastTime: Optional[float] = None

    def addLetter(self, letter: str, t: Optional[float] = None) -> str:
        t = t or time.time()
        self.lastTime = t
        self.buffer.append(letter.upper())
        return "".join(self.buffer)

    def shouldFlush(self) -> bool:
        return bool(
            self.buffer
            and self.lastTime is not None
            and time.time() - self.lastTime >= WORD_GAP
        )

    def wordConfidence(self, word: str, maxLen: int = 12) -> float:
        if word not in self.wordSet:
            return 0.0
        return 0.6 + 0.4 * min(len(word) / maxLen, 1.0)

    def autocorrect(self, word: str) -> Optional[str]:
        matches = get_close_matches(word, self.wordSet, n=1, cutoff=AUTOCORRECT_THRESHOLD)
        return matches[0] if matches else None

    def flush(self):
        word      = "".join(self.buffer)
        corrected = self.autocorrect(word) if AUTOCORRECT_TOGGLE else None
        if corrected:
            result, conf, tag = corrected, self.wordConfidence(corrected), "auto"
        else:
            conf = self.wordConfidence(word)
            if conf >= CONFIDENCE_THRESHOLD:
                result, tag = word, "word"
            else:
                result, tag = word, "letters"
        self.buffer.clear()
        self.lastTime = None
        return result, conf, tag

    def clear(self) -> None:
        self.buffer.clear()
        self.lastTime = None

    def deleteLast(self) -> str:
        if self.buffer:
            self.buffer.pop()
        return "".join(self.buffer)


# ── WhisperWorker ─────────────────────────────────────────────────────────

class WhisperWorker(qtc.QThread):
    textReady  = qtc.pyqtSignal(str)
    logMessage = qtc.pyqtSignal(str, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.currentChunkDuration = INITIAL_CHUNK_DURATION
        self.lastText  = ""
        self.running   = True
        try:
            self.mic = sc.default_microphone()
        except Exception as e:
            print(f"[WARNING] No microphone found: {e}")
            self.mic = None
        self.diarization_pipeline = None

    def stop(self) -> None:
        self.running = False

    def transcribeAudio(self, audio: np.ndarray) -> str:
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
        buffer.seek(0)
        segments, _ = self.model.transcribe(buffer, beam_size=5)
        return "\n".join(seg.text for seg in segments)

    def run(self) -> None:
        self.logMessage.emit("Loading Whisper model...", LogLevel.INFO)
        self.model = WhisperManager.getModel()
        self.logMessage.emit("Whisper model loaded", LogLevel.INFO)
        self.diarization_pipeline = None  # diarization removed - requires gated HuggingFace token
        recorded = np.zeros((0, 1), dtype=np.float32)
        while self.running:
            if self.mic is None:
                self.logMessage.emit(
                    "No microphone found — audio transcription disabled",
                    LogLevel.WARNING
                )
                return
            with self.mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
                chunk = recorder.record(numframes=int(self.currentChunkDuration * SAMPLE_RATE))
            recorded = np.concatenate([recorded, chunk], axis=0)
            recorded = recorded[-(SAMPLE_RATE * 30):]
            text = self.transcribeAudio(recorded)
            if self.currentChunkDuration > MIN_CHUNK_DURATION:
                self.currentChunkDuration = max(
                    self.currentChunkDuration - CHUNK_DECREMENT, MIN_CHUNK_DURATION
                )
            if text != self.lastText:
                self.textReady.emit(text.strip())
                self.lastText = text


# ── TTSWorker ─────────────────────────────────────────────────────────────

class TTSWorker(threading.Thread):
    """Speaks text on a dedicated background thread so the GUI never blocks."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name="TTSWorker")
        self._q: queue.Queue = queue.Queue()
        self.available = False
        self._use_say = False

    def run(self) -> None:
        import sys
        engine = None

        try:
            import pyttsx3
            engine = pyttsx3.init()
            self.available = True
        except Exception as e:
            print(f"[WARNING] pyttsx3 init failed: {e}")

        if not self.available and sys.platform == "darwin":
            import subprocess
            try:
                subprocess.run(["say", ""], timeout=5, capture_output=True, check=False)
                self._use_say = True
                self.available = True
            except Exception as e:
                print(f"[WARNING] macOS 'say' fallback failed: {e}")

        if not self.available:
            return

        while True:
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                if self._use_say:
                    import subprocess
                    subprocess.run(["say", text], timeout=30, check=False)
                else:
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                print(f"[WARNING] TTS speak error: {e}")

    def speak(self, text: str) -> None:
        if self.available:
            self._q.put(text)

    def stop(self) -> None:
        self._q.put(None)



# ── AspectRatioWidget ─────────────────────────────────────────────────────

class AspectRatioWidget(qtw.QWidget):
    def __init__(self, ratio: float = 16/9, parent=None) -> None:
        super().__init__(parent)
        self.ratio  = ratio
        self.pixmap = None
        self.label  = qtw.QLabel(self)
        self.label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            qtw.QSizePolicy.Policy.Expanding,
            qtw.QSizePolicy.Policy.Expanding
        )
        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

    def setPixmap(self, pixmap: qtg.QPixmap) -> None:
        self.pixmap = pixmap
        if pixmap is not None:
            self.label.setPixmap(pixmap.scaled(
                self.label.size(),
                qtc.Qt.AspectRatioMode.KeepAspectRatio,
                qtc.Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event) -> None:
        if self.pixmap:
            self.label.setPixmap(self.pixmap.scaled(
                self.label.size(),
                qtc.Qt.AspectRatioMode.KeepAspectRatio,
                qtc.Qt.TransformationMode.SmoothTransformation
            ))

    def paintEvent(self, event) -> None:
        if not self.pixmap:
            return
        painter  = qtg.QPainter(self)
        painter.setRenderHint(qtg.QPainter.RenderHint.SmoothPixmapTransform)
        widget_w = self.width()
        widget_h = self.height()
        target_w = widget_w
        target_h = int(target_w / self.ratio)
        if target_h > widget_h:
            target_h = widget_h
            target_w = int(target_h * self.ratio)
        x = (widget_w - target_w) // 2
        y = (widget_h - target_h) // 2
        scaled = self.pixmap.scaled(
            target_w, target_h,
            qtc.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            qtc.Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(qtc.QRect(x, y, target_w, target_h), scaled)
        painter.end()


# ── GestureRecognizerWithoutLinesWorker ───────────────────────────────────

class GestureRecognizerWithoutLinesWorker(qtc.QObject):
    gestureRecognized = qtc.pyqtSignal(str, float)
    landmarksReady    = qtc.pyqtSignal(object, tuple)

    def __init__(
        self,
        classifier_path: Path,
        label_path: Path,
        landmark_path: Path,
        parent=None
    ) -> None:
        super().__init__(parent)
        self.enabled               = False
        self.clf                   = None
        self.labels: List[str]     = []
        self.mp_hands              = None
        self.hand_tracking_enabled = False
        self.lastProcessTime       = 0.0
        self.minInterval           = 0.15
        self.lastEmitTime          = 0.0
        self.lastGesture           = None
        self._stabilityLabel       = None
        self._stabilityCount       = 0
        self._prevVec              = None   # motion gate: previous frame's position vector

        # Load sklearn classifier
        try:
            boot_log("loading sklearn classifier...")
            self.clf     = joblib.load(str(classifier_path))
            boot_log("sklearn classifier loaded")
            self.enabled = True
            print(f"[INFO] Classifier loaded from {classifier_path}")
        except FileNotFoundError:
            print(f"[WARNING] Classifier not found at {classifier_path}. Train a model first.")

        # Load labels
        try:
            self.labels = label_path.read_text().splitlines()
            print(f"[INFO] Loaded {len(self.labels)} labels")
        except FileNotFoundError:
            print(f"[WARNING] Labels not found at {label_path}.")

        # Load MediaPipe HandLandmarker
        try:
            if not landmark_path.exists():
                print(f"[WARNING] hand_landmarker.task not found at {landmark_path}")
                print("[INFO] Run scripts/setup.py to download it.")
            else:
                base_options = python.BaseOptions(model_asset_path=str(landmark_path))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=1,
                    min_hand_detection_confidence=0.2,
                    min_hand_presence_confidence=0.2,
                    min_tracking_confidence=0.7,
                    running_mode=vision.RunningMode.IMAGE,
                )
                boot_log("loading MediaPipe HandLandmarker...")
                self.mp_hands              = vision.HandLandmarker.create_from_options(options)
                boot_log("MediaPipe HandLandmarker loaded")
                self.hand_tracking_enabled = True
                print("[INFO] HandLandmarker initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize HandLandmarker: {e}")
            self.mp_hands              = None
            self.hand_tracking_enabled = False

    @qtc.pyqtSlot(np.ndarray)
    def processFrame(self, frame: np.ndarray) -> None:
        if not self.enabled or self.clf is None or not self.hand_tracking_enabled:
            return
        now = time.time()
        if now - self.lastProcessTime < self.minInterval:
            return
        if frame is None:
            return
        self.lastProcessTime = now

        frame_flipped = cv2.flip(frame, 1)
        rgb           = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        mp_image      = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result        = self.mp_hands.detect(mp_image)

        if not result.hand_landmarks:
            return

        lm    = result.hand_landmarks[0]
        wrist = lm[0]
        ref   = lm[9]  # middle-finger MCP — used as scale reference
        scale = float(np.sqrt(
            (ref.x - wrist.x)**2 + (ref.y - wrist.y)**2 + (ref.z - wrist.z)**2
        ))
        if scale < 1e-6:
            return
        vec_flat = np.array(
            [[(l.x - wrist.x) / scale, (l.y - wrist.y) / scale, (l.z - wrist.z) / scale]
             for l in lm],
            dtype=np.float32
        ).flatten()

        # Motion gate: skip frames where the hand is moving too fast (waving,
        # transitioning orientation).  Preserves the current stability count so
        # a brief movement doesn't wipe accumulated progress on a held sign.
        if self._prevVec is not None:
            _motion = float(np.linalg.norm(vec_flat - self._prevVec))
            self._prevVec = vec_flat.copy()
            if _motion > 2.0:
                return
        else:
            self._prevVec = vec_flat.copy()
        _joints = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19),
                   (2,3,4),(6,7,8),(10,11,12),(14,15,16),(18,19,20)]
        _angles = []
        for _a, _b, _c in _joints:
            _v1 = np.array([lm[_a].x-lm[_b].x, lm[_a].y-lm[_b].y, lm[_a].z-lm[_b].z])
            _v2 = np.array([lm[_c].x-lm[_b].x, lm[_c].y-lm[_b].y, lm[_c].z-lm[_b].z])
            _cos = np.dot(_v1, _v2) / (np.linalg.norm(_v1) * np.linalg.norm(_v2) + 1e-6)
            _angles.append(float(np.clip(_cos, -1.0, 1.0)))
        _thumb = np.array([
            (lm[4].x - wrist.x) / scale,
            (lm[4].y - wrist.y) / scale,
            (lm[4].z - wrist.z) / scale,
        ])
        _dists = []
        for _t in [8, 12, 16, 20]:   # index, middle, ring, pinky tips
            _tip = np.array([
                (lm[_t].x - wrist.x) / scale,
                (lm[_t].y - wrist.y) / scale,
                (lm[_t].z - wrist.z) / scale,
            ])
            _dists.append(float(np.linalg.norm(_tip - _thumb)))
        _tips = [
            np.array([(lm[i].x-wrist.x)/scale, (lm[i].y-wrist.y)/scale, (lm[i].z-wrist.z)/scale])
            for i in [8, 12, 16, 20]
        ]
        _adj = [
            float(np.linalg.norm(_tips[0] - _tips[1])),
            float(np.linalg.norm(_tips[1] - _tips[2])),
            float(np.linalg.norm(_tips[2] - _tips[3])),
        ]
        # Z-depth differentials: index-middle and middle-ring crossing depth.
        # Directly encodes which finger is "in front" — key for R (index behind
        # middle) vs U (same depth) vs V (same depth, wider spread).
        _zdiff = [
            float(_tips[0][2] - _tips[1][2]),   # index_tip.z - middle_tip.z
            float(_tips[1][2] - _tips[2][2]),   # middle_tip.z - ring_tip.z
        ]

        _n_feat = getattr(self.clf, 'n_features_in_', 80)
        if _n_feat >= 82:
            vec = np.concatenate([
                vec_flat,
                np.array(_angles,   dtype=np.float32),
                np.array(_dists,    dtype=np.float32),
                np.array(_adj,      dtype=np.float32),
                np.array(_zdiff,    dtype=np.float32),
            ]).reshape(1, 82)
        else:
            vec = np.concatenate([
                vec_flat,
                np.array(_angles,   dtype=np.float32),
                np.array(_dists,    dtype=np.float32),
                np.array(_adj,      dtype=np.float32),
            ]).reshape(1, 80)

        try:
            proba = self.clf.predict_proba(vec)[0]
        except ValueError:
            # Feature count mismatch — model was trained with different features.
            # Retrain the model to fix this.
            return
        idx   = int(np.argmax(proba))
        score = float(proba[idx])
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        if not label or idx >= len(self.labels):
            return

        self.landmarksReady.emit(result.hand_landmarks[0], frame_flipped.shape[:2])

        # Sticky stability window: a wrong frame costs 1 point instead of
        # resetting to zero.  This lets R survive the occasional U frame that
        # MediaPipe emits when tracking crossed fingers, without lowering the
        # bar for genuinely ambiguous or transitional poses.
        if label == self._stabilityLabel:
            self._stabilityCount += 1
        else:
            self._stabilityCount -= 1
            if self._stabilityCount <= 0:
                self._stabilityLabel = label
                self._stabilityCount = 1

        threshold = SETTINGS.settings.confidence_threshold
        if (score > threshold
                and self._stabilityCount >= 4
                and now - self.lastEmitTime > 1.0):
            self.lastEmitTime = now
            self.lastGesture  = label
            self.gestureRecognized.emit(label, score)
            self._stabilityCount = 0

    @qtc.pyqtSlot()
    def resetStability(self) -> None:
        """Reset the stability accumulator so transition frames don't carry over."""
        self._stabilityCount = 0
        self._stabilityLabel = None
        self._prevVec        = None

    def reload(self) -> None:
        """Reload classifier and labels from disk after new training."""
        try:
            self.clf     = joblib.load(str(CLASSIFIER_MODEL_PATH))
            self.enabled = True
            print("[INFO] Classifier reloaded.")
        except FileNotFoundError:
            print("[WARNING] Classifier not found — train first.")
        try:
            self.labels = LABELS_PATH.read_text().splitlines()
            print(f"[INFO] Labels reloaded: {self.labels}")
        except FileNotFoundError:
            print("[WARNING] Labels file not found.")


# ══════════════════════════════════════════════════════════════════════════
# MainGui
# ══════════════════════════════════════════════════════════════════════════

class MainGui(qtw.QMainWindow):
    frameForGesture      = qtc.pyqtSignal(np.ndarray)
    resetWorkerStability = qtc.pyqtSignal()

    def __init__(self, startup_warnings: Optional[List[str]] = None) -> None:
        boot_log("__init__ start")
        super().__init__()

        # Apply stylesheet
        try:
            from ui.style import STYLESHEET
            self.setStyleSheet(STYLESHEET)
        except ImportError:
            pass  # stylesheet is optional — app works without it

        self.title = SETTINGS.app.name
        self.setWindowTitle(self.title)
        self.windowManager = WindowManager(self, SETTINGS)
        self.windowManager.apply(
            SETTINGS.app.fullscreen_mode,
            SETTINGS.app.width,
            SETTINGS.app.height
        )
        self.move(self.windowManager.posx, self.windowManager.posy)

        self.runtimeLogger    = UILogger("runtime")
        self.translatorLogger = UILogger("translator")
        self.workerLogger     = UILogger("worker")

        # Load word list
        self.wordSet = set()
        if not WORDLIST.exists():
            raise FileNotFoundError(f"Word list not found: {WORDLIST}")
        with open(WORDLIST, encoding="utf-8") as f:
            self.wordSet = {line.strip().upper() for line in f if line.strip()}

        self.decoder    = WordDecoder(self.wordSet)
        self.logLevel   = LogLevel.INFO
        self.modelDir   = Path(DATASET_PATH)
        self.exportDir  = Path(EXPORT_PATH)
        self.datasetPath = Path(DATASET_PATH)
        self.exportPath  = Path(EXPORT_PATH)
        self.exampleAmount  = NUM_EXAMPLES
        self.workerLogPath  = Path(WORKER_LOG_PATH)
        self.lines          = SETTINGS.settings.lines
        self.capturing      = False
        self._lastCapTime   = 0.0
        self.currentGesture = None
        self._latestLandmarks  = None
        self._latestFrameShape = None

        self.logStatus("Loaded Word List", LogLevel.INFO)

        # Gesture worker thread
        self.gestureThread = qtc.QThread(self)
        self.signRecognizerNoLines = GestureRecognizerWithoutLinesWorker(
            CLASSIFIER_MODEL_PATH, LABELS_PATH, LANDMARK_MODEL_PATH
        )
        self.signRecognizerNoLines.moveToThread(self.gestureThread)
        self.gestureThread.finished.connect(self.signRecognizerNoLines.deleteLater)
        self.gestureThread.start()
        self.logStatus("Loaded Model Detection", LogLevel.INFO)

        # Worker log polling
        self._logFilePos      = 0
        self.lastWorkerLogLine = 0
        self.workerLogTimer    = qtc.QTimer()
        self.workerLogTimer.timeout.connect(self.readWorkerLogs)
        self.workerLogTimer.start(250)

        # Layout scaffolding
        self.cameraViewLayout          = qtw.QVBoxLayout()
        self.translatorCameraViewLayout = qtw.QVBoxLayout()
        self.statusLayout              = qtw.QVBoxLayout()
        self.translatorStatusLayout    = qtw.QVBoxLayout()

        self.windowManager.applyDPI()
        os.makedirs(self.modelDir, exist_ok=True)

        # Camera widgets (created before tabs so tabs can reference them)
        self.translatorCameraView = AspectRatioWidget(16/9)
        self.cameraView           = AspectRatioWidget(16/9)
        self.cameraViewLayout.addWidget(self.cameraView)
        self.translatorCameraView.setSizePolicy(
            qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.cameraView.setSizePolicy(
            qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)

        # Central widget
        self.outLayout  = qtw.QVBoxLayout()
        self.centralWid = qtw.QWidget()
        self.centralWid.setLayout(self.outLayout)
        self.setCentralWidget(self.centralWid)

        # Log viewers (created before tabs)
        self.statusOutput           = LogViewer(maxLines=1500)
        self.translatorStatusOutput = LogViewer(maxLines=1500)
        self.workerOutput           = LogViewer(maxLines=1500)

        self.runtimeLogger.logReady.connect(self.statusOutput.enqueue)
        self.translatorLogger.logReady.connect(self.translatorStatusOutput.enqueue)
        self.workerLogger.logReady.connect(self.statusOutput.enqueue)

        self.frame = None

        # Build tabs
        self.tabs = qtw.QTabWidget()
        self.tabs.addTab(self.translatorTabUI(), "Translator")
        self.tabs.addTab(self.modelMakerTabUI(), "Model Maker")
        self.tabs.addTab(self.settingsTabUI(), "Settings")

        self.tabs.currentChanged.connect(lambda _: self.updateFrame())
        self.tabs.currentChanged.connect(self.onTabChanged)
        self.outLayout.addWidget(self.tabs, 0)

        # Camera
        boot_log("initCamera start")
        self.initCamera()
        boot_log("initCamera done")
        self.gestures = self.syncGesturesWithFilesystem()
        self.loadExistingGestures(orderByName=True)

        # Timers
        self.frameTimer = qtc.QTimer()
        self.frameTimer.timeout.connect(self.updateFrame)
        self.frameTimer.start(33)

        self.wordTimer = qtc.QTimer()
        self.wordTimer.timeout.connect(self.checkWordBoundary)
        self.wordTimer.start(200)

        # Whisper worker
        self.whisperWorker = WhisperWorker()
        self.whisperWorker.textReady.connect(self.updateTranscription)
        self.whisperWorker.logMessage.connect(self.logStatus)

        # TTS worker
        self.ttsEnabled = True
        self.ttsWorker = TTSWorker()
        self.ttsWorker.start()
        qtc.QTimer.singleShot(3000, self._checkTTSAvailable)

        # Signal connections
        self.frameForGesture.connect(self.signRecognizerNoLines.processFrame)
        self.signRecognizerNoLines.gestureRecognized.connect(self.updateASLTranscription)
        self.signRecognizerNoLines.landmarksReady.connect(self._receiveLandmarks)
        self.resetWorkerStability.connect(self.signRecognizerNoLines.resetStability)

        self.letterBuffer              = []
        self.lastGestureTime           = None
        self.gapBetweenSignRecognition = 1.0
        self.gestureInterval           = 0.12
        self._lastGestureTime          = 0.0
        self._lastAcceptedGesture      = None
        self._lastAcceptedTime         = 0.0
        self._lastFrameEmitTime        = 0.0
        self._committedWords           = []   # list of (word, tag, separator)

        if self.cap:
            boot_log("launching camera thread")
            self.launchCameraThread()
            boot_log("camera thread launched")
            self.updateFrame()

        # Show startup warnings in log
        if startup_warnings:
            for w in startup_warnings:
                self.logStatus(f"WARNING: {w}", LogLevel.WARNING)
        boot_log("__init__ done")

    # ── Tab builders ──────────────────────────────────────────────────────

    def translatorTabUI(self) -> qtw.QWidget:
        self.translatorTab    = qtw.QWidget()
        layout                = qtw.QGridLayout()
        self.scoreTranscriptionOutput = qtw.QTextEdit()
        self.scoreTranscriptionOutput.setReadOnly(True)
        self.transcriptionOutput = qtw.QTextEdit()
        self.transcriptionOutput.setReadOnly(True)
        self.aslTranscriptionOutput = qtw.QTextEdit()
        self.aslTranscriptionOutput.setReadOnly(True)
        self.audioRecordBtn = qtw.QPushButton("Record Audio")
        self.audioRecordBtn.setCheckable(True)
        self.audioRecordBtn.clicked.connect(self.toggleAudioRecording)

        dataLabel = qtw.QLabel(
            f"Log: {WORKER_LOG_PATH}\nModel: {MODEL_PATH}"
        )
        dataLabel.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextSelectableByMouse)
        dataLabel.setStyleSheet("color: white; font-style: italic; font-size: 11px;")

        self.translatorCameraViewLayout.addWidget(qtw.QLabel("Signing View"), 0)
        self.translatorCameraViewLayout.addWidget(self.translatorCameraView, 1)
        layout.addLayout(self.translatorCameraViewLayout, 0, 0, 2, 1)

        outputLayout = qtw.QVBoxLayout()
        outputLayout.addWidget(qtw.QLabel("Signed Output"))

        aslCtrlRow = qtw.QHBoxLayout()
        clearBtn = qtw.QPushButton("Clear")
        clearBtn.setToolTip("Clear all signed output")
        clearBtn.clicked.connect(self.clearTranscript)
        exportBtn = qtw.QPushButton("Export")
        exportBtn.setToolTip("Save transcript to file")
        exportBtn.clicked.connect(self.exportTranscript)
        addSpaceBtn = qtw.QPushButton("Commit Word  [Space]")
        addSpaceBtn.setToolTip("Commit current letters as a word (same as pressing Space)")
        addSpaceBtn.clicked.connect(lambda: self._manualFlush(append=" "))
        aslCtrlRow.addWidget(clearBtn)
        aslCtrlRow.addWidget(exportBtn)
        aslCtrlRow.addWidget(addSpaceBtn)
        outputLayout.addLayout(aslCtrlRow)

        outputLayout.addWidget(self.aslTranscriptionOutput, 2)
        outputLayout.addWidget(qtw.QLabel("Audio Transcription"))
        outputLayout.addWidget(self.transcriptionOutput, 1)
        layout.addLayout(outputLayout, 0, 2, 3, 1)

        layout.addWidget(self.audioRecordBtn, 1, 1)

        self.ttsSpeakBtn = qtw.QPushButton("Speak Words: ON")
        self.ttsSpeakBtn.setCheckable(True)
        self.ttsSpeakBtn.setChecked(True)
        self.ttsSpeakBtn.clicked.connect(self.toggleTTS)
        layout.addWidget(self.ttsSpeakBtn, 2, 1)

        self.translatorStatusFrame = qtw.QWidget()
        self.translatorStatusLayout.addWidget(qtw.QLabel("Debug Log"), 0)
        self.translatorStatusLayout.addWidget(self.translatorStatusOutput, 1)
        self.translatorStatusFrame.setLayout(self.translatorStatusLayout)
        layout.addWidget(self.translatorStatusFrame, 2, 0, 1, 1)
        layout.addWidget(dataLabel, 4, 0, 1, -1)

        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(2, 2)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        self.translatorTab.setLayout(layout)
        return self.translatorTab

    def modelMakerTabUI(self) -> qtw.QWidget:
        self.modelMakerTab    = qtw.QWidget()
        mainLayout            = qtw.QGridLayout()

        # Toolbar
        self.startCaptureBtn  = qtw.QPushButton("Start Capture")
        self.startCaptureBtn.setCheckable(True)
        self.startCaptureBtn.clicked.connect(self.startCapture)
        mainLayout.addWidget(self.startCaptureBtn, 0, 0)

        refreshBtn = qtw.QPushButton("Refresh")
        refreshBtn.clicked.connect(self.refreshGestures)
        mainLayout.addWidget(refreshBtn, 0, 1)

        self.visualizeModelBtn = qtw.QPushButton("Visualize Model")
        self.visualizeModelBtn.clicked.connect(self.visualizeModel)
        mainLayout.addWidget(self.visualizeModelBtn, 0, 2)

        self.trainExportModelBtn = qtw.QPushButton("Train and Export Model")
        self.trainExportModelBtn.clicked.connect(self.trainExportModel)
        mainLayout.addWidget(self.trainExportModelBtn, 0, 3)

        reloadBtn = qtw.QPushButton("Reload Model")
        reloadBtn.clicked.connect(self.reloadGestureModel)
        mainLayout.addWidget(reloadBtn, 0, 4)

        folderBtn = qtw.QPushButton("Open Folder")
        folderBtn.clicked.connect(self.openVersionFolder)
        mainLayout.addWidget(folderBtn, 0, 5)

        self.quitProgramBtn = qtw.QPushButton("Quit Program")
        self.quitProgramBtn.clicked.connect(self.close)
        mainLayout.addWidget(self.quitProgramBtn, 0, 6)

        # Gesture management
        gestureLayout  = qtw.QVBoxLayout()
        btnLayout      = qtw.QHBoxLayout()
        self.gestureNameInput = qtw.QLineEdit()
        self.gestureNameInput.setPlaceholderText("New Gesture Name")
        self.gestureNameInput.returnPressed.connect(self.gestureNameExistsCheck)
        btnLayout.addWidget(self.gestureNameInput)
        addBtn = qtw.QPushButton("Add")
        addBtn.clicked.connect(self.gestureNameExistsCheck)
        btnLayout.addWidget(addBtn)
        delBtn = qtw.QPushButton("Delete")
        delBtn.clicked.connect(self.gestureSelectedCheck)
        btnLayout.addWidget(delBtn)
        gestureLayout.addLayout(btnLayout)

        self.listGesturesTree = qtw.QTreeWidget()
        self.listGesturesTree.setHeaderHidden(True)
        self.listGesturesTree.setRootIsDecorated(False)
        self.listGesturesTree.setDragEnabled(True)
        self.listGesturesTree.setAcceptDrops(True)
        self.listGesturesTree.setDropIndicatorShown(True)
        self.listGesturesTree.setDragDropMode(
            qtw.QAbstractItemView.DragDropMode.InternalMove)
        gestureLayout.addWidget(self.listGesturesTree)

        self.gestureTreeInfo = qtw.QTreeWidget()
        self.gestureTreeInfo.setColumnCount(2)
        self.gestureTreeInfo.setHeaderLabels(["Gesture", "Images"])
        self.gestureTreeInfo.header().setSectionResizeMode(
            qtw.QHeaderView.ResizeMode.Stretch)
        gestureLayout.addWidget(self.gestureTreeInfo)

        gestureWidget = qtw.QWidget()
        gestureWidget.setLayout(gestureLayout)
        gestureWidget.setFixedWidth(260)
        mainLayout.addWidget(gestureWidget, 1, 0)

        # Camera
        mainLayout.addLayout(self.cameraViewLayout, 1, 1, 1, 3)

        # Training status
        statusLayout    = qtw.QVBoxLayout()
        self.statusFrame = qtw.QWidget()

        self.trainingProgressBar = qtw.QProgressBar()
        self.trainingProgressBar.setRange(0, 100)
        self.trainingProgressBar.setValue(0)
        self.trainingProgressBar.setFormat("Ready")
        statusLayout.addWidget(self.trainingProgressBar)

        accRow = qtw.QHBoxLayout()
        accRow.addWidget(qtw.QLabel("Last Accuracy:"))
        self.accuracyLabel = qtw.QLabel("—")
        accRow.addWidget(self.accuracyLabel)
        statusLayout.addLayout(accRow)

        self.accuracyBar = qtw.QProgressBar()
        self.accuracyBar.setRange(0, 100)
        self.accuracyBar.setValue(0)
        self.accuracyBar.setFixedHeight(8)
        self.accuracyBar.setTextVisible(False)
        statusLayout.addWidget(self.accuracyBar)

        statusLayout.addWidget(qtw.QLabel("Training Log:"))
        statusLayout.addWidget(self.statusOutput)

        self.statusLayout.addWidget(self.statusOutput, 1)
        self.statusFrame.setLayout(self.statusLayout)
        mainLayout.addWidget(self.statusFrame, 3, 0, 1, -1)

        pathLabel = qtw.QLabel(f"Dataset: {DATASET_PATH}")
        pathLabel.setStyleSheet("color: white; font-style: italic; font-size: 11px;")
        pathLabel.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextSelectableByMouse)
        mainLayout.addWidget(pathLabel, 4, 0, 1, -1)

        self.modelMakerTab.setLayout(mainLayout)
        return self.modelMakerTab

    def settingsTabUI(self) -> qtw.QWidget:
        self.settingsTab    = qtw.QWidget()
        layout              = qtw.QGridLayout()

        self.cameraMenu     = qtw.QComboBox()
        self.resolutionMenu = qtw.QComboBox()
        self.windowModeMenu = qtw.QComboBox()
        self.monitorMenu    = qtw.QComboBox()
        self.aspectRatioScaleMenu = qtw.QComboBox()
        self.dpiCheck       = qtw.QCheckBox("Enable DPI Scaling")
        self.debugCheckbox  = qtw.QCheckBox("Enable debug logging")
        self.visualizeModelExamplesInput = qtw.QLineEdit()
        self.resetSettingsButton = qtw.QPushButton("Reset All Settings to Defaults")

        self.dpiCheck.setChecked(SETTINGS.app.dpi_scaling)

        for i, screen in enumerate(qtw.QApplication.screens()):
            s = screen.size()
            self.monitorMenu.addItem(f"Monitor {i}: {s.width()}x{s.height()}", i)
        self.monitorMenu.setCurrentIndex(SETTINGS.app.monitor)
        self.monitorMenu.currentIndexChanged.connect(self.changeMonitor)

        self.aspectRatioScaleMenu.addItems(["16:9", "4:3"])
        self.windowModeMenu.addItems(["Windowed", "Fullscreen", "Borderless Fullscreen"])
        self.windowModeMenu.setCurrentText(self.windowManager.mode.title())

        self.windowModeMenu.currentTextChanged.connect(self.changeWindowMode)
        self.aspectRatioScaleMenu.currentTextChanged.connect(self.updateWindowResolutions)
        self.resolutionMenu.currentIndexChanged.connect(self.updateWindowSizeValues)
        self.dpiCheck.stateChanged.connect(self.toggleDPIScaling)
        self.cameraMenu.currentIndexChanged.connect(
            lambda: ConfigAPI.update("app", "camera", self.cameraMenu.currentData())
        )
        self.updateWindowResolutions(self.aspectRatioScaleMenu.currentText())

        self.sampleRateInput = qtw.QSpinBox()
        self.sampleRateInput.setRange(8000, 48000)
        self.sampleRateInput.setValue(SETTINGS.settings.sam_rate)

        self.initialChunkDerationInput = qtw.QDoubleSpinBox()
        self.initialChunkDerationInput.setRange(1.0, 30.0)
        self.initialChunkDerationInput.setValue(SETTINGS.settings.init_chunk_der)

        self.minimumChunkDerationInput = qtw.QDoubleSpinBox()
        self.minimumChunkDerationInput.setRange(0.5, 10.0)
        self.minimumChunkDerationInput.setValue(SETTINGS.settings.min_chunk_der)

        self.chunkDecrementInput = qtw.QDoubleSpinBox()
        self.chunkDecrementInput.setRange(0.1, 5.0)
        self.chunkDecrementInput.setValue(SETTINGS.settings.chunk_dec)

        self.confidenceThresholdInput = qtw.QDoubleSpinBox()
        self.confidenceThresholdInput.setRange(0.0, 1.0)
        self.confidenceThresholdInput.setSingleStep(0.05)
        self.confidenceThresholdInput.setValue(SETTINGS.settings.confidence_threshold)

        self.autocorrectThresholdInput = qtw.QDoubleSpinBox()
        self.autocorrectThresholdInput.setRange(0.5, 1.0)
        self.autocorrectThresholdInput.setSingleStep(0.05)
        self.autocorrectThresholdInput.setValue(SETTINGS.settings.autocorrect_threshold)

        self.wordGapInput = qtw.QDoubleSpinBox()
        self.wordGapInput.setRange(0.3, 5.0)
        self.wordGapInput.setSingleStep(0.1)
        self.wordGapInput.setValue(SETTINGS.settings.word_gap)

        self.linesCheckBoxInput   = qtw.QCheckBox("Show landmark lines")
        self.linesCheckBoxInput.setChecked(SETTINGS.settings.lines)
        self.AutocorrectToggleInput = qtw.QCheckBox("Enable autocorrect")
        self.AutocorrectToggleInput.setChecked(SETTINGS.settings.autocorrect)
        self.previewToggleInput   = qtw.QCheckBox("Show live letter preview")
        self.previewToggleInput.setChecked(SETTINGS.settings.preview_toggle)
        self.confidenceToggleInput = qtw.QCheckBox("Show confidence scores")
        self.confidenceToggleInput.setChecked(SETTINGS.settings.confidence_toggle)

        self.logLevelInput = qtw.QComboBox()
        self.logLevelInput.addItems(["Debug", "Info", "Warning", "Error"])
        self.logLevelInput.setCurrentIndex(SETTINGS.app.log_level)

        self.hfTokenInput = qtw.QLineEdit()
        self.hfTokenInput.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
        self.hfTokenInput.setEchoMode(qtw.QLineEdit.EchoMode.Password)
        self.hfTokenInput.setText(SETTINGS.env.hf_token)

        self.saveSettingsBtn = qtw.QPushButton("Save Settings")
        self.saveSettingsBtn.clicked.connect(self.updateSettings)

        self.widthInput  = None
        self.heightInput = None

        form = qtw.QFormLayout()
        form.addRow("Monitor",             self.monitorMenu)
        form.addRow("Camera",              self.cameraMenu)
        form.addRow("Window Mode",         self.windowModeMenu)
        form.addRow("Aspect Ratio",        self.aspectRatioScaleMenu)
        form.addRow("Resolution",          self.resolutionMenu)
        form.addRow("",                    self.dpiCheck)
        form.addRow("Sample Rate (Hz)",    self.sampleRateInput)
        form.addRow("Initial Chunk (s)",   self.initialChunkDerationInput)
        form.addRow("Minimum Chunk (s)",   self.minimumChunkDerationInput)
        form.addRow("Chunk Decrement",     self.chunkDecrementInput)
        form.addRow("Min Word Confidence", self.confidenceThresholdInput)
        form.addRow("Autocorrect Cutoff",  self.autocorrectThresholdInput)
        form.addRow("Word Gap (s)",        self.wordGapInput)
        form.addRow("",                    self.linesCheckBoxInput)
        form.addRow("",                    self.AutocorrectToggleInput)
        form.addRow("",                    self.previewToggleInput)
        form.addRow("",                    self.confidenceToggleInput)
        form.addRow("Log Level",           self.logLevelInput)
        form.addRow("HF Token",            self.hfTokenInput)
        form.addRow("",                    self.debugCheckbox)

        layout.addLayout(form, 0, 0)
        layout.addWidget(self.saveSettingsBtn, 1, 0)
        layout.addWidget(self.resetSettingsButton, 2, 0)

        self.debugCheckbox.stateChanged.connect(self.toggleDebugLogging)
        self.resetSettingsButton.pressed.connect(self.confirmResetSettings)
        self.monitorMenu.setCurrentIndex(self.windowManager.monitorIndex)

        self.settingsTab.setLayout(layout)
        return self.settingsTab

    # ── Camera ────────────────────────────────────────────────────────────

    def initCamera(self) -> None:
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            return
        self.cap = None
        for i in range(4):
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            tmp = cv2.VideoCapture(i, backend)
            if tmp.isOpened():
                ret, _ = tmp.read()
                if ret:
                    self.cap = tmp
                    self.cameraIndex = i
                    break
                tmp.release()
        if not self.cap or not self.cap.isOpened():
            self.errorMenu(message="No camera found")
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ratio    = actual_w / actual_h if actual_h > 0 else 16/9
        self.translatorCameraView.ratio = ratio
        self.cameraView.ratio           = ratio
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
        fps   = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAMERA] format={codec} fps={fps} size={actual_w}x{actual_h}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.logStatus(f"Camera: {actual_w}x{actual_h}", LogLevel.INFO)
        self.stopEvent  = threading.Event()
        self.frameLock  = threading.Lock()
        self.frame      = None
        self.cameraThread = None

    def cameraLoop(self) -> None:
        while not getattr(self, "stopEvent", threading.Event()).is_set() and self.cap:
            ret, frame = self.cap.read()
            if ret:
                with self.frameLock:
                    self.frame = frame

    def launchCameraThread(self) -> None:
        if getattr(self, "cameraThread", None) and self.cameraThread.is_alive():
            return
        self.stopEvent.clear()
        self.cameraThread = threading.Thread(target=self.cameraLoop, daemon=True)
        self.cameraThread.start()
        self.logStatus("Camera started", LogLevel.INFO)

    def stopCamera(self) -> None:
        self.frameTimer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def startCamera(self) -> None:
        if not self.cap or not self.cap.isOpened():
            preferred = SETTINGS.app.camera
            working   = findWorkingCamera(preferred) or findWorkingCamera(0)
            if working is None:
                raise RuntimeError("No working camera detected.")
            self.cap = cv2.VideoCapture(working)
        self.frameTimer.start(33)

    # ── Frame processing ──────────────────────────────────────────────────

    def updateFrame(self) -> None:
        if hasattr(self, "settingsTab") and self.tabs.currentWidget() is self.settingsTab:
            return
        if not hasattr(self, "frameLock"):
            return
        frame_copy = None
        with self.frameLock:
            if self.frame is None:
                return
            frame_copy = self.frame.copy()

        if self.currentGesture:
            cv2.putText(frame_copy, self.currentGesture, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 100), 2)

        if self.capturing and self.currentGesture:
            cv2.circle(frame_copy, (30, 30), 12, (0, 0, 220), -1)
            cv2.putText(frame_copy, f"REC {self.currentGesture}",
                        (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

        rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg   = qtg.QImage(rgb.data, w, h, ch * w, qtg.QImage.Format.Format_RGB888)
        pixmap = qtg.QPixmap.fromImage(qimg)

        if self.lines and self._latestLandmarks:
            pixmap = self._applyLandmarkOverlay(pixmap)

        if self.cameraView.isVisible():
            self.cameraView.setPixmap(pixmap)
        if self.translatorCameraView.isVisible():
            now_emit = time.time()
            if now_emit - self._lastFrameEmitTime >= 0.15:
                self._lastFrameEmitTime = now_emit
                self.frameForGesture.emit(frame_copy)
            self.translatorCameraView.setPixmap(pixmap)

        if self.capturing and self.currentGesture:
            now = time.time()
            if now - self._lastCapTime >= 0.5:
                self._lastCapTime = now
                gesture_dir = self.modelDir / self.currentGesture
                gesture_dir.mkdir(parents=True, exist_ok=True)
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                filename = gesture_dir / f"{ts}.jpg"
                frame_to_save = cv2.flip(frame_copy, 1)
                cv2.imwrite(str(filename), frame_to_save)

    # ── Landmark overlay ──────────────────────────────────────────────────

    @qtc.pyqtSlot(object, tuple)
    def _receiveLandmarks(self, landmarks, frame_shape: tuple) -> None:
        self._latestLandmarks  = landmarks
        self._latestFrameShape = frame_shape

    def _applyLandmarkOverlay(self, pixmap: qtg.QPixmap) -> qtg.QPixmap:
        if not self._latestLandmarks:
            return pixmap
        result  = pixmap.copy()
        painter = qtg.QPainter(result)
        painter.setRenderHint(qtg.QPainter.RenderHint.Antialiasing)
        pw, ph  = pixmap.width(), pixmap.height()

        def lm_to_px(lm):
            return int(lm.x * pw), int(lm.y * ph)

        pen = qtg.QPen(qtg.QColor("#00D4AA"), 2)
        painter.setPen(pen)
        for a, b in HAND_CONNECTIONS:
            lms = self._latestLandmarks
            if a < len(lms) and b < len(lms):
                ax, ay = lm_to_px(lms[a])
                bx, by = lm_to_px(lms[b])
                painter.drawLine(ax, ay, bx, by)

        painter.setPen(qtg.QPen(qtg.QColor("#FFFFFF"), 1))
        painter.setBrush(qtg.QBrush(qtg.QColor("#00D4AA")))
        for lm in self._latestLandmarks:
            x, y = lm_to_px(lm)
            painter.drawEllipse(qtc.QPoint(x, y), 4, 4)

        painter.end()
        return result

    # ── ASL transcription ─────────────────────────────────────────────────

    @qtc.pyqtSlot(str, float)
    def updateASLTranscription(self, name: str, score: float) -> None:
        SPECIAL = {"DEL", "DELETE", "BACKSPACE"}
        now = time.time()
        self._lastAcceptedGesture = name
        self._lastAcceptedTime    = now
        if name.upper() in SPECIAL:
            preview = self.decoder.deleteLast()
        else:
            preview = self.decoder.addLetter(name)
        if SETTINGS.settings.preview_toggle:
            self._refreshASLDisplay(preview)

    def _refreshASLDisplay(self, preview: str = "") -> None:
        committed = "".join(w + sep for w, _, sep in self._committedWords)
        self.aslTranscriptionOutput.setPlainText(committed + preview)
        cursor = self.aslTranscriptionOutput.textCursor()
        cursor.movePosition(qtg.QTextCursor.MoveOperation.End)
        self.aslTranscriptionOutput.setTextCursor(cursor)
        self.aslTranscriptionOutput.ensureCursorVisible()

    def _mergeAndCommit(self, text: str, tag: str, separator: str) -> None:
        """
        Append a committed word, attempting to merge it with preceding letter-fragments
        if the concatenation forms a dictionary word.  Only 'letters'-tagged entries
        (partial words not found in the dictionary) are ever merged — matched words,
        autocorrected words, and entries separated by a newline are left untouched.
        """
        if tag == "letters":
            candidates = []
            for i in range(len(self._committedWords) - 1, -1, -1):
                w, t, sep = self._committedWords[i]
                if sep == "\n" or t != "letters":
                    break
                candidates.insert(0, i)
                if len(candidates) >= 3:
                    break

            if candidates:
                parts = [self._committedWords[i][0] for i in candidates] + [text]
                for combo_len in range(len(parts), 1, -1):
                    combo = "".join(parts[-combo_len:]).upper()
                    if len(combo) <= 15 and combo in self.decoder.wordSet:
                        del self._committedWords[-(combo_len - 1):]
                        self._committedWords.append((combo, "merged", separator))
                        return

        self._committedWords.append((text, tag, separator))

    def checkWordBoundary(self) -> None:
        if self.decoder.shouldFlush():
            text, conf, tag = self.decoder.flush()
            if self.ttsEnabled:
                self.ttsWorker.speak(text)
            self._mergeAndCommit(text, tag, " ")
            self._refreshASLDisplay()
            self.resetWorkerStability.emit()

    def _manualFlush(self, append: str = " ") -> None:
        """Immediately commit the current letter buffer (hotkey / button trigger)."""
        if not self.decoder.buffer:
            return
        text, conf, tag = self.decoder.flush()
        if self.ttsEnabled:
            self.ttsWorker.speak(text)
        self._mergeAndCommit(text, tag, append)
        self._refreshASLDisplay()
        self.resetWorkerStability.emit()

    def updateTranscription(self, text: str) -> None:
        ts     = datetime.now().strftime("[%H:%M:%S] ")
        cursor = self.transcriptionOutput.textCursor()
        cursor.movePosition(qtg.QTextCursor.MoveOperation.End)
        cursor.insertText(ts + text.strip() + "\n")
        self.transcriptionOutput.setTextCursor(cursor)
        self.transcriptionOutput.ensureCursorVisible()

    def scoreTranscription(self, score: float, name: str) -> None:
        ts     = datetime.now().strftime("[%H:%M:%S] ")
        cursor = self.scoreTranscriptionOutput.textCursor()
        cursor.movePosition(qtg.QTextCursor.MoveOperation.End)
        cursor.insertText(f"{ts}{name} ({score:.2f})\n")
        self.scoreTranscriptionOutput.setTextCursor(cursor)
        self.scoreTranscriptionOutput.ensureCursorVisible()

    # ── TTS ───────────────────────────────────────────────────────────────

    def toggleTTS(self) -> None:
        self.ttsEnabled = self.ttsSpeakBtn.isChecked()
        self.ttsSpeakBtn.setText("Speak Words: ON" if self.ttsEnabled else "Speak Words: OFF")

    def _checkTTSAvailable(self) -> None:
        if not self.ttsWorker.available:
            self.ttsEnabled = False
            self.ttsSpeakBtn.setChecked(False)
            self.ttsSpeakBtn.setEnabled(False)
            self.ttsSpeakBtn.setText("Speak Words: unavailable")
            self.logStatus("TTS unavailable — no audio output device found", LogLevel.WARNING)

    # ── Audio recording ───────────────────────────────────────────────────

    def toggleAudioRecording(self) -> None:
        if self.whisperWorker.isRunning():
            self.whisperWorker.stop()
            self.whisperWorker.wait()
            self.audioRecordBtn.setText("Record Audio")
        else:
            self.whisperWorker.running = True
            self.whisperWorker.start()
            self.audioRecordBtn.setText("Stop Recording")

    # ── Transcript export / clear ─────────────────────────────────────────

    def exportTranscript(self) -> None:
        EXPORT_PATH.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = EXPORT_PATH / f"transcript_{ts}.txt"
        lines = [
            "=== ASL Interpreter Transcript ===",
            f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "", "--- Signed Output ---",
            self.aslTranscriptionOutput.toPlainText(),
            "", "--- Audio Transcription ---",
            self.transcriptionOutput.toPlainText(),
        ]
        out_file.write_text("\n".join(lines), encoding="utf-8")
        self.logStatus(f"Transcript saved: {out_file}", LogLevel.INFO)
        os.startfile(str(out_file))

    def clearTranscript(self) -> None:
        self._committedWords = []
        self.aslTranscriptionOutput.clear()
        self.transcriptionOutput.clear()
        self.decoder.clear()
        self.logStatus("Transcript cleared", LogLevel.INFO)

    # ── Model training ────────────────────────────────────────────────────

    def reloadGestureModel(self) -> None:
        self.signRecognizerNoLines.reload()
        self.logStatus("Gesture model reloaded", LogLevel.INFO)

    def runTraining(self) -> None:
        train_script = _REPO_ROOT / "src/app/scripts/train.py"
        self.trainingProcess = qtc.QProcess(self)
        self.trainingProcess.setProgram(sys.executable)
        self.trainingProcess.setArguments([
            str(train_script),
            "--dataset", str(DATASET_PATH),
            "--output",  str(_REPO_ROOT / "src/deploy"),
            "--logfile", str(WORKER_LOG_PATH),
        ])
        self.trainingProcess.readyReadStandardOutput.connect(self._onTrainingOutput)
        self.trainingProcess.readyReadStandardError.connect(self._onTrainingError)
        self.trainingProcess.finished.connect(self._onTrainingFinished)
        WORKER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        WORKER_LOG_PATH.write_text("")
        self._logFilePos       = 0
        self.lastWorkerLogLine = 0
        self.trainingProcess.start()

    def _onTrainingOutput(self) -> None:
        out = bytes(self.trainingProcess.readAllStandardOutput()).decode()
        self.logStatus(out.strip(), LogLevel.INFO)

    def _onTrainingError(self) -> None:
        err = bytes(self.trainingProcess.readAllStandardError()).decode()
        self.logStatus(err.strip(), LogLevel.ERROR)

    def _onTrainingFinished(self, code: int, _) -> None:
        self.trainExportModelBtn.setEnabled(True)
        if code == 0:
            self.trainingProgressBar.setValue(100)
            self.trainingProgressBar.setFormat("Complete")
            self.signRecognizerNoLines.reload()
            self._updateAccuracyDisplay()
            self.logStatus("Training finished — model reloaded.", LogLevel.INFO)
        else:
            self.trainingProgressBar.setFormat("Failed")
            self.logStatus(f"Training failed (exit {code})", LogLevel.ERROR)

    def _updateAccuracyDisplay(self) -> None:
        meta_path = _REPO_ROOT / "src/deploy/model_meta.json"
        try:
            meta = json.loads(meta_path.read_text())
            acc  = meta.get("accuracy", 0)
            self.accuracyLabel.setText(f"{acc:.1%}")
            self.accuracyBar.setValue(int(acc * 100))
        except Exception:
            pass

    def trainExportModel(self) -> None:
        # Warn about low image counts
        data    = self.loadData()
        MINIMUM = 15
        low     = [(d["name"], d["image_count"]) for d in data if d["image_count"] < MINIMUM]
        if low:
            msg = "\n".join(f"  {n}: {c} images" for n, c in low)
            reply = qtw.QMessageBox.warning(
                self, "Low Image Count",
                f"These gestures have fewer than {MINIMUM} images:\n{msg}\n\nContinue?",
                qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No,
                qtw.QMessageBox.StandardButton.No
            )
            if reply == qtw.QMessageBox.StandardButton.No:
                return
        self.logStatus("Starting training...", LogLevel.INFO)
        self.trainingProgressBar.setValue(0)
        self.trainingProgressBar.setFormat("Starting...")
        self.trainExportModelBtn.setEnabled(False)
        self._trainingTotal = 0
        self._trainingDone  = 0
        self.runTraining()

    # ── Worker log polling ────────────────────────────────────────────────

    def readWorkerLogs(self) -> None:
        try:
            lines     = WORKER_LOG_PATH.read_text(encoding="utf-8").splitlines()
            new_lines = lines[self.lastWorkerLogLine:]
            self.lastWorkerLogLine = len(lines)
        except FileNotFoundError:
            return

        for line in new_lines:
            level, msg = self.parseWorkerLogLine(line)
            self.workerLogger.log(msg, level)

            # Progress tracking
            if any(c in line for c in ("✓", "⚠", "✗")):
                if hasattr(self, "_trainingTotal") and self._trainingTotal > 0:
                    self._trainingDone = getattr(self, "_trainingDone", 0) + 1
                    pct = min(int(self._trainingDone / self._trainingTotal * 90), 90)
                    self.trainingProgressBar.setValue(pct)
                    self.trainingProgressBar.setFormat(f"Training... {pct}%")

            if "Found labels" in line:
                import re
                m = re.search(r'\[(.+?)\]', line)
                if m:
                    self._trainingTotal = len(m.group(1).split(","))
                    self._trainingDone  = 0

            if "ACCURACY:" in line:
                try:
                    acc = float(line.split("ACCURACY:")[1].strip())
                    self.accuracyLabel.setText(f"{acc:.1%}")
                    self.accuracyBar.setValue(int(acc * 100))
                except (ValueError, IndexError):
                    pass

    def parseWorkerLogLine(self, line: str):
        if "[ERROR]"   in line: return LogLevel.ERROR,   line.strip()
        if "[WARNING]" in line: return LogLevel.WARNING, line.strip()
        if "[DEBUG]"   in line: return LogLevel.DEBUG,   line.strip()
        return LogLevel.INFO, line.strip()

    # ── Gesture management ────────────────────────────────────────────────

    def syncGesturesWithFilesystem(self) -> list:
        DATASET_PATH.mkdir(parents=True, exist_ok=True)
        folders = sorted(
            [d for d in DATASET_PATH.iterdir() if d.is_dir()],
            key=lambda d: d.name.lower()
        )
        data = [{"name": d.name, "image_count": self.countImages(d)} for d in folders]
        self.saveData(data)
        self.logStatus(f"Synced {len(data)} gestures from filesystem", LogLevel.DEBUG)
        return data

    def loadData(self) -> list:
        if not os.path.exists(JSON_FILE):
            return []
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        return data

    def saveData(self, data) -> None:
        with open(JSON_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def countImages(self, directory) -> int:
        if not os.path.exists(directory):
            return 0
        return sum(1 for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS))

    def addGesture(self, name: str) -> None:
        data = self.loadData()
        if any(e["name"] == name for e in data):
            self.logStatus(f"'{name}' already exists.")
            return
        image_dir = Path(DATASET_PATH) / name
        image_dir.mkdir(parents=True, exist_ok=True)
        data.append({"name": name, "image_count": self.countImages(image_dir)})
        self.saveData(data)
        self.loadExistingGestures()
        self.gestureNameInput.clear()
        self.logStatus(f"Added '{name}'", LogLevel.INFO)

    def updateAllImageCounts(self) -> None:
        data = self.loadData()
        for entry in data:
            folder = os.path.join(DATASET_PATH, entry["name"])
            entry["image_count"] = self.countImages(folder)
        self.saveData(data)

    def deleteGesture(self, name: str) -> None:
        self.frameTimer.stop()
        self.listGesturesTree.setCurrentItem(None)
        self.gestureTreeInfo.setCurrentItem(None)
        self.currentGesture = None
        data = self.loadData()
        if hasattr(self, "stopEvent"):
            self.stopEvent.set()
        if self.cameraThread:
            self.cameraThread.join(timeout=0.2)
        data = [d for d in data if d["name"] != name]
        self.saveData(data)
        gesture_dir = Path(DATASET_PATH) / name
        if gesture_dir.exists():
            shutil.rmtree(gesture_dir)
        self.logStatus(f"Deleted '{name}'", LogLevel.INFO)
        self.currentGesture = None
        self.capturing      = False
        self.frameTimer.start(33)
        self.launchCameraThread()
        self.loadExistingGestures()

    def loadExistingGestures(self, orderByName: bool = True) -> list:
        self.listGesturesTree.clear()
        self.gestureTreeInfo.clear()
        data = self.syncGesturesWithFilesystem()
        if orderByName:
            data = sorted(data, key=lambda x: x["name"].lower())
        self.gestures = data
        for entry in data:
            name  = entry["name"]
            count = self.countImages(DATASET_PATH / name)
            self.listGesturesTree.addTopLevelItem(qtw.QTreeWidgetItem([name]))
            self.gestureTreeInfo.addTopLevelItem(qtw.QTreeWidgetItem([name, str(count)]))
        return data

    def refreshGestures(self) -> None:
        self.loadExistingGestures(orderByName=True)
        self.logStatus("Refreshed gestures", LogLevel.INFO)

    def gestureNameExistsCheck(self) -> None:
        name = self.gestureNameInput.text().strip()
        if name:
            self.addGesture(name)
        else:
            self.errorMenu("The gesture does not have a name.")

    def gestureSelectedCheck(self) -> None:
        item = self.selectedGesture()
        if item:
            self.confirmGestureDelete(item)
        else:
            self.errorMenu("No gesture is selected.")

    def selectedGesture(self):
        item = self.listGesturesTree.currentItem()
        if not item:
            return None
        name = item.text(0)
        for i in range(self.gestureTreeInfo.topLevelItemCount()):
            info_item = self.gestureTreeInfo.topLevelItem(i)
            if info_item.text(0) == name:
                self.gestureTreeInfo.setCurrentItem(info_item)
                break
        return item

    def confirmGestureDelete(self, item) -> None:
        item = self.listGesturesTree.currentItem()
        if not item:
            return
        name  = item.text(0)
        reply = qtw.QMessageBox.question(
            self, "Confirm Deletion",
            f"Delete gesture '{name}'?",
            qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No,
            qtw.QMessageBox.StandardButton.No
        )
        if reply == qtw.QMessageBox.StandardButton.Yes:
            self.frameTimer.stop()
            if self.capturing:
                self.capturing = False
                self.startCaptureBtn.setText("Start Capture")
            self.deleteGesture(name)
            self.frameTimer.start(33)

    def startCapture(self) -> None:
        self.toggleCapture()

    def toggleCapture(self) -> None:
        if not self.cap:
            self.errorMenu("No camera available.")
            return
        item = self.listGesturesTree.currentItem()
        if not item:
            self.errorMenu("No gesture selected")
            return
        gesture_name    = item.text(0)
        self.currentGesture = gesture_name
        self.capturing      = not self.capturing
        if self.capturing:
            self.startCaptureBtn.setText("Stop Capture")
            os.makedirs(os.path.join(self.modelDir, gesture_name), exist_ok=True)
            self.logStatus(f"Capture started for '{gesture_name}'", LogLevel.INFO)
        else:
            self.startCaptureBtn.setText("Start Capture")
            self.updateAllImageCounts()
            self.refreshGestures()
        self.logStatus(
            f"Capture {'ON' if self.capturing else 'OFF'} for '{gesture_name}'",
            LogLevel.INFO
        )

    def visualizeModel(self) -> None:
        labels = [
            i for i in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, i))
        ]
        for label in labels:
            label_dir = os.path.join(DATASET_PATH, label)
            files     = os.listdir(label_dir)[:NUM_EXAMPLES]
            if not files:
                continue
            fig, axs = plt.subplots(1, min(NUM_EXAMPLES, len(files)), figsize=(10, 2))
            if len(files) == 1:
                axs = [axs]
            for i, fname in enumerate(files):
                axs[i].imshow(plt.imread(os.path.join(label_dir, fname)))
                axs[i].axis("off")
            fig.suptitle(f"{label} — {len(files)} examples")
        plt.show()

    def openVersionFolder(self) -> None:
        os.startfile(self.modelDir)

    # ── Settings ──────────────────────────────────────────────────────────

    def updateSettings(self) -> None:
        fullscreen_mode = {
            "Windowed":           "Windowed",
            "Fullscreen":         "Fullscreen",
            "Borderless Fullscreen": "Borderless",
        }.get(self.windowModeMenu.currentText(), "Windowed")

        examples_text = getattr(self, "visualizeModelExamplesInput", None)
        examples = (int(examples_text.text())
                    if (examples_text and examples_text.text().isdigit())
                    else NUM_EXAMPLES)

        ConfigAPI.bulkUpdate({
            "app": {
                "fullscreen_mode": fullscreen_mode,
                "log_level":       self.logLevelInput.currentIndex(),
                "monitor":         self.monitorMenu.currentData() or 0,
                "camera":          self.cameraMenu.currentData() or 0,
                "dpi_scaling":     self.dpiCheck.isChecked(),
            },
            "settings": {
                "examples":              examples,
                "sam_rate":              self.sampleRateInput.value(),
                "init_chunk_der":        self.initialChunkDerationInput.value(),
                "min_chunk_der":         self.minimumChunkDerationInput.value(),
                "chunk_dec":             self.chunkDecrementInput.value(),
                "lines":                 self.linesCheckBoxInput.isChecked(),
                "confidence_threshold":  self.confidenceThresholdInput.value(),
                "autocorrect":           self.AutocorrectToggleInput.isChecked(),
                "autocorrect_threshold": self.autocorrectThresholdInput.value(),
                "word_gap":              self.wordGapInput.value(),
                "preview_toggle":        self.previewToggleInput.isChecked(),
                "confidence_toggle":     self.confidenceToggleInput.isChecked(),
            }
        })

        token = self.hfTokenInput.text().strip()
        if token:
            ConfigAPI.setEnvToken(token)

        self.lines = self.linesCheckBoxInput.isChecked()
        new_level  = self.logLevelInput.currentIndex()
        self.runtimeLogger.setLevel(new_level)
        self.translatorLogger.setLevel(new_level)
        self.workerLogger.setLevel(new_level)
        self.changeWindowMode(self.windowModeMenu.currentText())
        self.logStatus("Settings saved.", LogLevel.INFO)

    def changeMonitor(self, i: int) -> None:
        self.windowManager.apply(monitor=i)
        self.updateWindowResolutions(self.aspectRatioScaleMenu.currentText())

    def toggleDPIScaling(self, state) -> None:
        self.windowManager.dpiScaling = bool(state)
        self.windowManager.applyDPI()

    def updateWindowSizeValues(self) -> None:
        if self.windowManager.mode != WindowMode.WINDOWED:
            return
        data = self.resolutionMenu.currentData()
        if data:
            w, h = data
            self.widthInput  = w
            self.heightInput = h
            self.windowManager.apply(width=w, height=h)

    def changeWindowMode(self, text: str) -> None:
        mode_map = {
            "Windowed":            WindowMode.WINDOWED,
            "Fullscreen":          WindowMode.FULLSCREEN,
            "Borderless Fullscreen": WindowMode.BORDERLESS,
        }
        self.windowManager.apply(mode=mode_map.get(text, WindowMode.WINDOWED))

    def updateWindowResolutions(self, aspect: str) -> None:
        self.resolutionMenu.clear()
        for w, h in self.windowManager.availableResolutions():
            if aspect == "16:9" and abs((w/h) - (16/9)) > 0.01:
                continue
            if aspect == "4:3" and abs((w/h) - (4/3)) > 0.01:
                continue
            self.resolutionMenu.addItem(f"{w}x{h}", (w, h))

    def toggleDebugLogging(self, state) -> None:
        level = LogLevel.DEBUG if state else LogLevel.INFO
        self.runtimeLogger.setLevel(level)
        self.translatorLogger.setLevel(level)
        self.workerLogger.setLevel(level)

    def confirmResetSettings(self) -> None:
        reply = qtw.QMessageBox.question(
            self, "Confirm Reset",
            "Reset all settings to defaults?",
            qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No,
            qtw.QMessageBox.StandardButton.No
        )
        if reply == qtw.QMessageBox.StandardButton.Yes:
            loadDefaultSettings()

    def reloadSettings(self) -> None:
        global SETTINGS
        SETTINGS = loadSettings()

    # ── Utility ───────────────────────────────────────────────────────────

    def listAvailableCameras(self, max_tested: int = 3) -> List[int]:
        active = getattr(self, "cameraIndex", None)
        cams = []
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        for i in range(max_tested):
            if i == active:
                cams.append(i)
                continue
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                cams.append(i)
            cap.release()
        return cams

    def logStatus(self, message, level: int = LogLevel.INFO) -> None:
        self.runtimeLogger.log(message, level)
        self.translatorLogger.log(message, level)
        self.workerLogger.log(message, level)

    def errorMenu(self, message: str) -> None:
        qtw.QMessageBox.critical(self, "Error", message, qtw.QMessageBox.StandardButton.Ok)

    def onTabChanged(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if widget in (self.translatorTab, self.modelMakerTab):
            if not self.frameTimer.isActive():
                self.frameTimer.start(33)
        else:
            self.frameTimer.stop()
        if widget is self.settingsTab and self.cameraMenu.count() == 0:
            for i in self.listAvailableCameras():
                self.cameraMenu.addItem(f"Camera {i}", i)

    def keyPressEvent(self, e) -> None:
        key = e.key()
        if key == qtc.Qt.Key.Key_Escape and self.windowManager.mode != WindowMode.WINDOWED:
            self.windowManager.apply(mode=WindowMode.WINDOWED)
        elif key == qtc.Qt.Key.Key_Space:
            self._manualFlush(append=" ")
        elif key in (qtc.Qt.Key.Key_Return, qtc.Qt.Key.Key_Enter):
            self._manualFlush(append="\n")
        elif key == qtc.Qt.Key.Key_Backspace:
            preview = self.decoder.deleteLast()
            if SETTINGS.settings.preview_toggle:
                self._refreshASLDisplay(preview)
        elif key == qtc.Qt.Key.Key_Delete:
            self.decoder.clear()
            if SETTINGS.settings.preview_toggle:
                self._refreshASLDisplay("")

    def closeEvent(self, event) -> None:
        if hasattr(self, "stopEvent"):
            self.stopEvent.set()
        if hasattr(self, "cameraThread") and self.cameraThread:
            self.cameraThread.join(timeout=1)
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        if hasattr(self, "ttsWorker"):
            self.ttsWorker.stop()
        if hasattr(self, "whisperWorker"):
            self.whisperWorker.stop()
            self.whisperWorker.wait()
        if hasattr(self, "gestureThread"):
            self.gestureThread.quit()
            self.gestureThread.wait()
        self.windowManager.saveState()
        ConfigAPI.update("app", "width",          self.windowManager.width)
        ConfigAPI.update("app", "height",         self.windowManager.height)
        ConfigAPI.update("app", "pos_x",          self.windowManager.posx)
        ConfigAPI.update("app", "pos_y",          self.windowManager.posy)
        ConfigAPI.update("app", "fullscreen_mode", self.windowManager.mode)
        ConfigAPI.update("app", "monitor",        self.windowManager.monitorIndex)
        super().closeEvent(event)
        qtw.QApplication.quit()


# ══════════════════════════════════════════════════════════════════════════
# Startup health check
# ══════════════════════════════════════════════════════════════════════════

def check_environment() -> List[str]:
    errors   = []
    warnings = []
    if not WORDLIST.exists():
        errors.append(f"Word list not found:\n  {WORDLIST}")
    if not LABELS_PATH.exists():
        errors.append(f"Labels file not found:\n  {LABELS_PATH}")
    if not LANDMARK_MODEL_PATH.exists():
        warnings.append(
            f"hand_landmarker.task not found — run: python scripts/setup.py"
        )
    if not CLASSIFIER_MODEL_PATH.exists():
        warnings.append(
            "No trained gesture model — use Model Maker tab to train one."
        )
    if errors:
        app = qtw.QApplication.instance() or qtw.QApplication(sys.argv)
        qtw.QMessageBox.critical(
            None, "Setup Error",
            "Required files missing:\n\n" + "\n\n".join(errors)
        )
        sys.exit(1)
    return warnings


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    boot_log("entering main")
    app              = qtw.QApplication(sys.argv)
    startup_warnings = check_environment()
    window           = MainGui(startup_warnings=startup_warnings)
    window.show()
    boot_log("entering Qt event loop")
    sys.exit(app.exec())
