#!/usr/bin/env python3
from config.loader import loadSettings  # Load persisted app config at startup.
from config.loadDefaults import loadDefaultSettings  # Restore config defaults.
from config.writer import ConfigAPI  # Read/update config values on disk.

# Media + ML stack used by capture, inference, and diagnostics.
import mediapipe as mp  # Optional hand-landmark visualization helpers.
import os  # Filesystem utilities and Explorer launch.
from mediapipe.tasks import python  # MediaPipe task runtime bindings.
from mediapipe.tasks.python import vision  # Vision task namespace.
import matplotlib.pyplot as plt  # Dataset sample preview plots.
from datetime import datetime  # Timestamping logs and captured frames.

# Qt UI stack for widgets, timers, threads, signals, and drawing.
import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg

# General runtime/system dependencies.
from pathlib import Path  # Cross-platform path handling.
from faster_whisper import WhisperModel  # Speech-to-text model runtime.
from pyannote.audio import Pipeline  # Optional diarization pipeline check.
import soundcard as sc  # Microphone capture backend.
import soundfile as sf  # Writes temporary WAV buffers for Whisper.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # TFLite interpreter for gesture classification.
import numpy as np  # Frame/audio tensor operations.
import io  # In-memory buffer for audio serialization.
import threading  # Background camera thread + synchronization primitives.
import subprocess  # Launches training script via WSL.
import json  # Gesture metadata persistence format.
import shutil  # Recursive dataset folder deletion.
import time  # Timing logic for word boundaries and throttles.
import sys  # QApplication argv/exit plumbing.
from enum import IntEnum  # Strongly typed log level enum.
import cv2  # Camera capture + frame preprocessing.
from difflib import get_close_matches  # Dictionary autocorrect matching.

# -----------------------------------------------------------------------------
# Global configuration and runtime constants
# -----------------------------------------------------------------------------
# Keep absolute path construction centralized so runtime code can use simple
# constants without worrying about the process working directory.
CONFIG_FILE = Path(__file__).parent.parent.parent / "app/src/config/config.dev.toml"
SETTINGS = loadSettings()
HF_TOKEN = SETTINGS.env.hf_token
VERSION = SETTINGS.version.version
SHARED = Path(__file__).parent.parent.parent / "shared"
DB_FILE = Path(__file__).parent / "gestures.db"
DATASET_PATH = Path(__file__).parent.parent.parent / "shared/datasets"
EXPORT_PATH = Path(__file__).parent.parent.parent / "shared/exports"
WORKER_LOG_PATH = Path(__file__).parent.parent.parent / "shared/logs/worker.log.json"
MODEL_PATH = Path(__file__).parent.parent.parent / "deploy/asl_model.tflite"
WORDLIST = Path(__file__).parent.parent.parent / "deploy/words.txt"
LABELS_PATH = Path(__file__).parent.parent.parent / "deploy/labels.txt"
MODEL_NAME = SETTINGS.gestures.gesture_model
CAMERA_INDEX = 0
NUM_EXAMPLES = SETTINGS.settings.examples
CONFIG_PATH = None
JSON_FILE = Path(__file__).parent.parent.parent / "shared/gestures.json"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
BASE_DATA = {
    "name": "", "image_count": ""
}
SAMPLE_RATE = SETTINGS.settings.sam_rate
INITIAL_CHUNK_DURATION = SETTINGS.settings.init_chunk_der
MIN_CHUNK_DURATION = SETTINGS.settings.min_chunk_der
CHUNK_DECREMENT = SETTINGS.settings.chunk_dec
CONFIDENCE_THRESHOLD = SETTINGS.settings.confidence_threshold
WORD_GAP = SETTINGS.settings.word_gap
AUTOCORRECT_TOGGLE = SETTINGS.settings.autocorrect
AUTOCORRECT_THRESHOLD = SETTINGS.settings.autocorrect_threshold
LOG_LEVEL = SETTINGS.app.log_level

# Probe for a usable camera index so UI setup can fail early with a clear error.
def findWorkingCamera(start_index=0, max_tested=5):
    """Return first camera index that opens successfully, else None."""
    # Probe only a small index range for faster startup.
    for i in range(start_index, max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

class WindowMode:
    WINDOWED = "windowed"
    FULLSCREEN = "fullscreen"
    BORDERLESS = "borderless"

class WindowManager:
    def __init__(self, window, settings):
        """Store window/settings state used to apply runtime display changes."""
        self.window = window
        self.settings = settings

        self.mode = settings.app.fullscreen_mode
        self.width = settings.app.width
        self.height = settings.app.height
        self.monitorIndex = settings.app.monitor
        self.posx = settings.app.pos_x
        self.posy = settings.app.pos_y
        self.dpiScaling = settings.app.dpi_scaling

    def screens(self):
        """Return Qt-visible monitors."""
        return qtw.QApplication.screens()
    
    def currentScreen(self):
        """Return selected monitor, clamped to valid range."""
        screens = self.screens()
        # Clamp to valid range in case monitor count changed since last run.
        return screens[min(self.monitorIndex, len(screens)-1)]
    
    def availableResolutions(self):
        """List common resolutions that fit on the selected monitor."""
        screen = self.currentScreen()
        size = screen.size()
        w, h = size.width(), size.height()
        # Generate common window sizes that fit on the selected monitor.
        common = [
            (3840,2160),(2560,1440),(2048,1536),(1920,1440),(1920,1080),
            (1600,900),(1400,1050),(1280,720),(1280,960),
            (1024,768),(800,600),(640,480)
        ]
        resolutions = sorted([(cw,ch) for cw,ch in common if cw<=w and ch<=h], reverse=True) 
        return resolutions

    def apply(self, mode=None, width=None, height=None, monitor=None):
        """Apply mode/size/monitor changes to the main window immediately."""
        if mode is not None:
            self.mode = mode
        if width:
            self.width = width
        if height:
            self.height = height
        if monitor is not None:
            self.monitorIndex = monitor

        w = self.window
        screen = self.currentScreen()

        # Reset to standard window flags before applying target mode.
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

    def saveState(self):
        """Persist current geometry fields from the live Qt window."""
        g = self.window.geometry()
        self.posx = g.x()
        self.posy = g.y()
        self.width = g.width()
        self.height = g.height()

    def applyDPI(self):
        """Apply high-DPI rounding policy when DPI scaling is enabled."""
        if self.dpiScaling:
            qtw.QApplication.setHighDpiScaleFactorRoundingPolicy(
                qtc.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )

class UILogger(qtc.QObject):
    logReady = qtc.pyqtSignal(str)

    def __init__(self, name="app", level=LogLevel.INFO):
        """Create a Qt-signal logger that emits formatted log lines."""
        super().__init__()
        self.name = name
        self.level = level 

    def setLevel(self, level):
        """Set active minimum log level for this logger."""
        self.level = level

    def log(self, message, level):
        """Emit a timestamped log message if level passes the threshold."""
        # Filter here once so all connected UI sinks receive the same stream.
        if level < self.level:
            return
        if level == LogLevel.DEBUG:
            levelText = "Debug"
        elif level == LogLevel.INFO:
            levelText = "Info"
        elif level == LogLevel.WARNING:
            levelText = "Warning"
        else:
            levelText = "Error"
        ts = datetime.now().strftime("[%H:%M:%S]")
        self.logReady.emit(f"{levelText} {ts} {message}")

class LogViewer(qtw.QTextEdit):
    def __init__(self, maxLines=1500, parent=None):
        """Create a buffered text log view with capped history."""
        super().__init__(parent)
        self.setReadOnly(True)
        self.maxLines = maxLines
        self.lines = []
        self.flushTimer = qtc.QTimer(self)
        self.flushTimer.timeout.connect(self.flush)
        self.flushTimer.start(100)  # Batch log UI updates (10 FPS).
        self.pending = []

    def enqueue(self, message):
        """Queue a log line for batched UI flush."""
        self.pending.append(message)

    def flush(self):
        """Flush pending lines into the widget and trim to max history."""
        if not self.pending:
            return
        self.lines.extend(self.pending)
        self.pending.clear()
        if len(self.lines) > self.maxLines:
            self.lines = self.lines[-self.maxLines:]
        self.setPlainText("\n".join(self.lines))
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class WhisperManager:
    _model = None

    @classmethod
    def getModel(cls):
        """Return shared singleton Whisper model instance."""
        # Lazily load once; Whisper initialization is expensive.
        if cls._model is None:
            print("Loading Whisper model...")
            cls._model = WhisperModel("small", device="cpu", compute_type="int8")
            print("Model loaded.")
        return cls._model

class WordDecoder:
    def __init__(self, wordSet):
        """Track recognized letters and convert them into words over time."""
        self.wordSet = wordSet
        self.buffer = []
        self.lastTime = None

    def push(self, letter, timestamp):
        """Append one letter with an explicit timestamp."""
        self.buffer.append(letter)
        self.lastTime = timestamp

    def addLetter(self, letter, t=None):
        """Append one letter and return current preview text."""
        t = t or time.time()
        self.lastTime = t
        self.buffer.append(letter)
        return " ".join(self.buffer)  # preview

    def shouldFlush(self):
        """Return True when buffered letters should be finalized into a token."""
        # If enough time passed since last letter, finalize the current token.
        return (
            self.buffer and
            self.lastTime and
            time.time() - self.lastTime >= WORD_GAP
        )

    def wordConfidence(self, word, maxLen=12):
        """Score words higher if valid and sufficiently long."""
        if word not in self.wordSet:
            return 0.0
        return 0.6 + 0.4 * min(len(word) / maxLen, 1.0)

    def autocorrect(self, word):
        """Return nearest dictionary word above fixed similarity cutoff."""
        matches = get_close_matches(word, self.wordSet, n=1, cutoff=0.85)
        return matches[0] if matches else None

    def flush(self):
        """Finalize buffered letters into word/letters output with confidence."""
        # Output preference:
        # 1) autocorrected dictionary word
        # 2) direct dictionary word above threshold
        # 3) spaced letters fallback when confidence is low
        word = "".join(self.buffer)
        corrected = self.autocorrect(word)

        if corrected:
            result = corrected
            conf = self.wordConfidence(corrected)
            tag = "auto"
        else:
            conf = self.wordConfidence(word)
            if conf >= CONFIDENCE_THRESHOLD:
                result = word
                tag = "word"
            else:
                result = " ".join(self.buffer)
                tag = "letters"
        self.buffer.clear()
        self.lastTime = None
        return result, conf, tag

class WhisperWorker(qtc.QThread):
    textReady = qtc.pyqtSignal(str)
    logMessage = qtc.pyqtSignal(str, int)
    def __init__(self, parent=None):
        """Initialize audio recorder and transcription dependencies."""
        super().__init__(parent)
        self.currentChunkDuration = INITIAL_CHUNK_DURATION
        self.lastText = ""
        self.running = True
        self.mic = sc.default_microphone()
        self.model = WhisperManager.getModel()
        try:
            Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        except Exception as e:
            self.logMessage.emit(f"Diarization disabled: {e}", LogLevel.WARNING)
            self.diarization_pipeline = None

    def stop(self):
        """Request the worker loop to stop."""
        self.running = False

    def transcribeAudio(self, audio):
        """Convert numpy audio to WAV buffer and transcribe with Whisper."""
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
        buffer.seek(0)
        segments, _ = self.model.transcribe(buffer, beam_size=5)
        return "\n".join(seg.text for seg in segments)

    def run(self):
        """Continuously capture microphone audio and emit incremental text."""
        self.logMessage.emit("Loading Whisper model...", LogLevel.INFO)
        recorded = np.zeros((0, 1), dtype=np.float32)
        while self.running:
            # Capture incrementally and keep a rolling 30s context to reduce resets.
            with self.mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
                chunk = recorder.record(numframes=int(self.currentChunkDuration * SAMPLE_RATE))
            recorded = np.concatenate([recorded, chunk], axis=0)
            maxSamples = SAMPLE_RATE * 30
            recorded = recorded[-maxSamples:]
            text = self.transcribeAudio(recorded)
            # Reduce chunk size gradually for lower latency after warm-up.
            if self.currentChunkDuration > MIN_CHUNK_DURATION:
                self.currentChunkDuration = max(self.currentChunkDuration - CHUNK_DECREMENT, MIN_CHUNK_DURATION)
            if text != self.lastText:
                # Emit only newly produced suffix to avoid duplicate text.
                newText = text[len(self.lastText):].strip()
                if newText:
                    self.textReady.emit(newText)
                self.lastText = text

class AspectRatioWidget(qtw.QWidget):
    def __init__(self, ratio=16/9, parent=None):
        """Display a pixmap while preserving a target aspect ratio."""
        super().__init__(parent)
        self.ratio = ratio
        self.pixmap = None
        self.label = qtw.QLabel(self)
        self.label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            qtw.QSizePolicy.Policy.Expanding,
            qtw.QSizePolicy.Policy.Expanding
        )
        self.layout = qtw.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.label)

    def setPixmap(self, pixmap: qtg.QPixmap):
        """Store and display incoming pixmap with aspect-preserving scaling."""
        self.pixmap = pixmap
        if pixmap is not None:
            self.label.setPixmap(pixmap.scaled(
                self.label.size(),
                qtc.Qt.AspectRatioMode.KeepAspectRatio,
                qtc.Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event):
        """Rescale current pixmap whenever widget size changes."""
        if self.pixmap:
            self.label.setPixmap(self.pixmap.scaled(
                self.label.size(),
                qtc.Qt.AspectRatioMode.KeepAspectRatio,
                qtc.Qt.TransformationMode.SmoothTransformation
            ))

    def paintEvent(self, event):
        """Draw centered, aspect-correct pixmap with smooth scaling."""
        if not self.pixmap:
            return
        self.painter = qtg.QPainter(self)
        self.painter.setRenderHint(qtg.QPainter.RenderHint.SmoothPixmapTransform)
        self.widget_w = self.width()
        self.widget_h = self.height()
        self.target_w = self.widget_w
        self.target_h = int(self.target_w / self.ratio)
        if self.target_h > self.widget_h:
            self.target_h = self.widget_h
            self.target_w = int(self.target_h * self.ratio)
        self.x = (self.widget_w - self.target_w) // 2
        self.y = (self.widget_h - self.target_h) // 2
        scaled = self.pixmap.scaled(
            self.target_w, self.target_h,
            qtc.Qt.AspectRatioMode.KeepAspectRatioByExpanding, qtc.Qt.TransformationMode.SmoothTransformation
        )
        self.painter.drawPixmap(
            qtc.QRect(self.x, self.y, self.target_w, self.target_h), scaled)
        self.painter.end()


class GestureRecognizerWithoutLinesWorker(qtc.QObject):
    gestureRecognized = qtc.pyqtSignal(str, float)

    def __init__(self, MODEL_PATH: str, LABEL_PATH: str, parent=None):
        """Load TFLite gesture model + labels and initialize inference state."""
        super().__init__(parent)

        self.modelPath = MODEL_PATH
        self.labels = open(LABEL_PATH).read().splitlines()

        self.running = False
        self.enabled = True
        self.lastProcessTime = 0.0
        self.minInterval = 0.10

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=self.modelPath)
        self.interpreter.allocate_tensors()

        # Input / Output info
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

        self.lastGesture = None
        self.lastEmitTime = 0
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    @qtc.pyqtSlot(np.ndarray)
    def processFrame(self, frame):
        now = time.time()

        if not self.enabled:
            return
        if now - self.lastProcessTime < self.minInterval:
            return
        if frame is None:
            return

        self.lastProcessTime = now

        # Mirror so on-screen motion feels natural
        frame = cv2.flip(frame, 1)

        # --- Step 1: Extract landmarks ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(rgb)

        # If no hand is visible in this frame, do nothing
        if not result.multi_hand_landmarks:
            return

        # --- Step 2: Build the 63-float input vector ---
        lm = result.multi_hand_landmarks[0].landmark
        wrist = lm[0]
        vec = np.array([
            [l.x - wrist.x, l.y - wrist.y, l.z - wrist.z]
            for l in lm
        ]).flatten().astype(np.float32).reshape(1, 63)

        # --- Step 3: Run TFLite inference ---
        self.interpreter.set_tensor(self.input_details[0]["index"], vec)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # --- Step 4: Emit result if confident enough ---
        idx = int(np.argmax(output))
        score = float(output[idx])
        label = self.labels[idx] if idx < len(self.labels) else str(idx)

        if score > 0.5:
            if now - self.lastEmitTime > 0.3:
                self.lastEmitTime = now
                self.lastGesture = label
                self.gestureRecognized.emit(label, score)

class MainGui(qtw.QMainWindow):
    frameForGesture = qtc.pyqtSignal(np.ndarray)
    def __init__(self):
        """Build the full GUI, start workers, and initialize camera pipelines."""
        super().__init__()
        # Window/bootstrap state.
        self.title = SETTINGS.app.name
        self.setWindowTitle(self.title)
        self.windowManager = WindowManager(self, SETTINGS)
        self.windowManager.apply(
            SETTINGS.app.fullscreen_mode,
            SETTINGS.app.width,
            SETTINGS.app.height
        )
        self.move(self.windowManager.posx, self.windowManager.posy)
        self.runtimeLogger = UILogger("runtime")
        self.translatorLogger = UILogger("translator")
        self.workerLogger = UILogger("worker")
        self.wordSet = set()
        if not WORDLIST.exists():
            raise FileNotFoundError(f"Word list not found: {WORDLIST}")
        with open(WORDLIST, encoding="utf-8") as f:
            self.wordSet = {line.strip().upper() for line in f if line.strip()}
        # Decoder turns streamed letters into final words based on timing.
        self.decoder = WordDecoder(self.wordSet)
        self.logStatus("Loaded Word List", LogLevel.INFO)
        self.logStatus(f"Window size: {self.windowManager.width}x{self.windowManager.height}", LogLevel.DEBUG)
        self.logLevel = LogLevel.INFO
        self.modelDir = Path(DATASET_PATH)
        self.exportDir = Path(EXPORT_PATH)
        self.datasetPath = Path(DATASET_PATH)
        self.exportPath = Path(EXPORT_PATH)
        self.exampleAmount = NUM_EXAMPLES
        self.workerLogPath = Path(WORKER_LOG_PATH)
        self.lines = SETTINGS.settings.lines
        # Run gesture inference in its own Qt thread to keep UI responsive.
        self.gestureThread = qtc.QThread(self)
        self.signRecognizerNoLines = GestureRecognizerWithoutLinesWorker(MODEL_PATH, LABELS_PATH)
        self.logStatus("Loaded Model Detection", LogLevel.INFO)
        self.signRecognizerNoLines.moveToThread(self.gestureThread)
        self.gestureThread.finished.connect(self.signRecognizerNoLines.deleteLater)
        self.gestureThread.start()
        self._logFilePos = 0
        self.lastWorkerLogLine = 0
        # Poll worker log file on timer so subprocess output appears live in UI.
        self.workerLogTimer = qtc.QTimer()
        self.workerLogTimer.timeout.connect(self.readWorkerLogs)
        self.workerLogTimer.start(250)
        self.cameraViewLayout = qtw.QVBoxLayout()
        self.translatorCameraViewLayout = qtw.QVBoxLayout()
        self.statusLayout = qtw.QVBoxLayout()
        self.translatorStatusLayout = qtw.QVBoxLayout()
        self.capturing = False
        self.currentGesture = None
        self.windowManager.applyDPI()
        os.makedirs(self.modelDir, exist_ok=True)
        self._logFilePos = 0
        self.workerLogTimer.start(250)
        self.translatorCameraView = AspectRatioWidget(16/9)
        self.cameraView = AspectRatioWidget(16/9)
        self.cameraViewLayout.addWidget(self.cameraView)
        self.outLayout = qtw.QVBoxLayout()
        self.translatorCameraView.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.cameraView.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.centralWid = qtw.QWidget()
        self.centralWid.setLayout(self.outLayout)
        self.setCentralWidget(self.centralWid)
        # Main tab shell + log panes.
        self.tabs = qtw.QTabWidget()
        self.quitTab = qtw.QWidget()
        self.statusOutput = LogViewer(maxLines=1500)
        self.translatorStatusOutput = LogViewer(maxLines=1500)
        self.workerOutput = LogViewer(maxLines=1500)
        self.runtimeLogger.logReady.connect(self.statusOutput.enqueue)
        self.translatorLogger.logReady.connect(self.translatorStatusOutput.enqueue)
        self.workerLogger.logReady.connect(self.statusOutput.enqueue)
        self.frame = None
        self.tabs.addTab(self.translatorTabUI(), "Translator")
        self.tabs.addTab(self.modelMakerTabUI(), "Model Maker")
        self.tabs.addTab(self.settingsTabUI(), "Settings")
        for i in self.listAvailableCameras():
            self.cameraMenu.addItem(f"Camera {i}", i)
        self.tabs.currentChanged.connect(lambda _: self.updateFrame())
        self.outLayout.addWidget(self.tabs, 0)
        self.initCamera()
        self.gestures = self.loadExistingGestures(orderByName=True)
        self.loadExistingGestures(orderByName=True)
        # Camera loop drives UI refresh at ~30 FPS.
        self.frameTimer = qtc.QTimer()
        self.frameTimer.timeout.connect(self.updateFrame)
        self.frameTimer.start(33)
        # Speech pipeline updates text asynchronously.
        self.whisperWorker = WhisperWorker()
        self.wordTimer = qtc.QTimer()
        self.wordTimer.timeout.connect(self.checkWordBoundary)
        self.wordTimer.start(200)
        self.whisperWorker.textReady.connect(self.updateTranscription)
        # Frame handoff to inference runs through a Qt signal for thread safety.
        self.frameForGesture.connect(self.signRecognizerNoLines.processFrame)
        self.letterBuffer = []
        self.lastGestureTime = None
        self.gapBetweenSignRecognition = 1.0
        self.gestureInterval = 0.12
        self._lastGestureTime = 0.0
        self.signRecognizerNoLines.gestureRecognized.connect(self.updateASLTranscription)
        self.whisperWorker.logMessage.connect(self.logStatus)
        if self.cap:
            self.launchCameraThread()
            self.updateFrame()
    
    def translatorTabUI(self):
        """Construct translator tab UI for live speech/sign outputs."""
        # This tab surfaces three streams:
        # microphone transcription, signed-letter decoding, and diagnostic logs.
        self.translatorTab = qtw.QWidget()
        self.translatorTabLayout = qtw.QGridLayout()
        self.scoreTranscriptionOutputLayout = qtw.QVBoxLayout()
        self.scoreTranscriptionOutput = qtw.QTextEdit()
        self.scoreTranscriptionOutput.setReadOnly(True)
        self.audioOutputTranscriptionLayout = qtw.QVBoxLayout()
        self.signedOutputTranscriptionLayout = qtw.QVBoxLayout()
        self.translatorCameraLabel = qtw.QLabel("Signing View Camera")
        self.translatorStatusOutputLabel = qtw.QLabel("Debug Information")
        self.audioOutputTranscriptionLabel = qtw.QLabel("Audio Output Transcription")
        self.signedOutputTranscriptionLabel = qtw.QLabel("Signed Output Transcription")
        self.scoreTranscriptionOutputLabel = qtw.QLabel("Signed Word Score")
        self.transcriptionOutput = qtw.QTextEdit()
        self.transcriptionOutput.setReadOnly(True)
        self.aslTranscriptionOutput = qtw.QTextEdit()
        self.aslTranscriptionOutput.setReadOnly(True)
        self.datasetPathLabel = qtw.QLabel(f"Log Path: {WORKER_LOG_PATH}\n"
                                           f"Current Model Loaded Model: {MODEL_PATH}")
        self.translatorStatusFrame = qtw.QWidget()
        self.audioRecordBtn = qtw.QPushButton("Record Audio")
        self.audioRecordBtnStatusLabel = qtw.QLabel
        self.translatorTabLayout.addLayout(self.audioOutputTranscriptionLayout, 1, 2)
        self.translatorTabLayout.addLayout(self.signedOutputTranscriptionLayout, 2, 2)
        self.scoreTranscriptionOutput.setText("…")
        self.datasetPathLabel.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.datasetPathLabel.setStyleSheet("color: white; font-style: italic; font-size: 14px;")
        self.scoreTranscriptionOutput.setStyleSheet("color: white; font-style: italic; font-size: 14px;")
        self.aslTranscriptionOutput.setStyleSheet("color: white; font-style: italic; font-size: 14px;")
        self.transcriptionOutput.setStyleSheet("color: white; font-style: italic; font-size: 14px;")
        self.translatorStatusOutput.setStyleSheet("color: white; font-style: italic; font-size: 14px;")
        self.datasetPathLabel.setToolTip("This is where gesture image data is stored")
        self.datasetPathLabel.mousePressEvent = lambda e: os.startfile(DATASET_PATH)
        self.datasetPathLabel.setCursor(qtc.Qt.CursorShape.PointingHandCursor)
        self.translatorCameraViewLayout.addWidget(self.translatorCameraLabel, 0)
        self.translatorCameraViewLayout.addWidget(self.translatorCameraView, 1)
        self.translatorTabLayout.addLayout(self.translatorCameraViewLayout, 0, 0, 2, 1)
        self.translatorTab.setLayout(self.translatorTabLayout)
        self.signedOutputTranscriptionLayout.addWidget(self.signedOutputTranscriptionLabel, 0)
        self.signedOutputTranscriptionLayout.addWidget(self.aslTranscriptionOutput, 1)
        self.audioOutputTranscriptionLayout.addWidget(self.audioOutputTranscriptionLabel, 0)
        self.audioOutputTranscriptionLayout.addWidget(self.transcriptionOutput, 1)
        self.scoreTranscriptionOutputLayout.addWidget(self.scoreTranscriptionOutputLabel, 0)
        self.scoreTranscriptionOutputLayout.addWidget(self.scoreTranscriptionOutput, 1)
        #self.translatorTabLayout.addLayout(self.scoreTranscriptionOutputLayout, 0, 2)
        self.translatorTabLayout.addWidget(self.audioRecordBtn, 1, 1)
        self.translatorStatusLayout.addWidget(self.translatorStatusOutputLabel, 0)
        self.translatorStatusLayout.addWidget(self.translatorStatusOutput, 1)
        self.translatorStatusFrame.setLayout(self.translatorStatusLayout)
        self.translatorTabLayout.addWidget(self.translatorStatusFrame, 2, 0, 1, 1)
        self.translatorTabLayout.addWidget(self.datasetPathLabel, 4, 0, 1, -1)
        self.audioRecordBtn.setCheckable(True)
        self.audioRecordBtn.clicked.connect(self.toggleAudioRecording)
        return self.translatorTab
    
    def modelMakerTabUI(self):
        """Construct model-maker tab for gesture dataset management/training."""
        #
        # Layouts: 
        # modelMakerTabLayout() is the main layout for the tab so it's just the outer ring for the tab 
        # outerGestureControlTreeBtnLayout() is the layout for the gesture management section of this tab organizes the whole gesture management section
        # gestureControlTreeBtnLayout() is the layout for the buttons that control the gesture management just used for convienece for organization
        # gestureControlTreeModelViewAndViewLayout() is the layout for showing what gesture exist and the information tied to them
        #
        #
        self.modelMakerTab = qtw.QWidget()
        self.modelMakerTabLayout = qtw.QGridLayout()
        self.treeAndCameraLayout = qtw.QHBoxLayout()
        self.outerGestureControlTreeBtnLayout = qtw.QVBoxLayout()
        self.gestureControlTreeBtnLayout = qtw.QHBoxLayout()
        self.gestureControlTreeModelViewAndViewLayout = qtw.QHBoxLayout()
        self.gestureNameInput = qtw.QLineEdit()
        self.gestureNameInput.setPlaceholderText("New Gesture Name: ")
        self.gestureControlTreeBtnLayout.addWidget(self.gestureNameInput, 0)
        self.addGestureBtn = qtw.QPushButton("Add")
        self.gestureControlTreeBtnLayout.addWidget(self.addGestureBtn, 1)
        self.addGestureBtn.clicked.connect(self.gestureNameExistsCheck)
        self.deleteGestureBtn = qtw.QPushButton("Delete Gesture",)
        self.gestureControlTreeBtnLayout.addWidget(self.deleteGestureBtn, 2)
        self.deleteGestureBtn.clicked.connect(self.gestureSelectedCheck)
        self.statusFrame = qtw.QWidget()
        self.startCaptureBtn = qtw.QPushButton("Start Capture")
        self.modelMakerTabLayout.addWidget(self.startCaptureBtn, 0, 0)
        self.startCaptureBtn.setCheckable(True)
        self.startCaptureBtn.clicked.connect(self.startCapture)
        self.refreshGesturesBtn = qtw.QPushButton("Refresh Gestures")
        self.modelMakerTabLayout.addWidget(self.refreshGesturesBtn, 0, 1)
        self.refreshGesturesBtn.clicked.connect(self.refreshGestures)
        self.visualizeModelBtn = qtw.QPushButton("Visualize Model")
        self.modelMakerTabLayout.addWidget(self.visualizeModelBtn, 0, 2)
        self.visualizeModelBtn.clicked.connect(self.visualizeModel)
        self.trainExportModelBtn = qtw.QPushButton("Train and Export Model")
        self.modelMakerTabLayout.addWidget(self.trainExportModelBtn, 0, 3)
        self.trainExportModelBtn.clicked.connect(self.trainExportModel)
        self.versionFolderBtn = qtw.QPushButton("Open Version Folder")
        self.modelMakerTabLayout.addWidget(self.versionFolderBtn, 0, 4)
        self.versionFolderBtn.clicked.connect(self.openVersionFolder)
        self.quitProgramBtn = qtw.QPushButton("Quit Program")
        self.modelMakerTabLayout.addWidget(self.quitProgramBtn, 0, 5)
        self.quitProgramBtn.clicked.connect(self.closeEvent)
        self.datasetPathLabel = qtw.QLabel(f"Image Storage Path: {DATASET_PATH}\n"
                                           f"Current Model: {self.modelDir}")
        self.datasetPathLabel.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.datasetPathLabel.setStyleSheet("color: white; font-style: italic; font-size: 14px;")
        self.datasetPathLabel.setToolTip("This is where gesture image data is stored")
        self.datasetPathLabel.mousePressEvent = lambda e: os.startfile(DATASET_PATH)
        self.datasetPathLabel.setCursor(qtc.Qt.CursorShape.PointingHandCursor)
        self.listGesturesTree = qtw.QTreeWidget()
        self.listGesturesTree.setHeaderHidden(True)
        self.gestureTreeInfo = qtw.QTreeWidget()
        self.gestureTreeInfo.setColumnCount(2)
        self.gestureTreeInfo.setHeaderLabels(["Gesture Name", "Image Count"])
        self.listGesturesTree.setDragEnabled(True)
        self.listGesturesTree.setAcceptDrops(True)
        self.listGesturesTree.setDropIndicatorShown(True)
        self.listGesturesTree.setDragDropMode(qtw.QAbstractItemView.DragDropMode.InternalMove)
        self.listGesturesTree.setRootIsDecorated(False)
        self.gestureControlTreeModelViewAndViewLayout.addWidget(self.listGesturesTree, 1)
        self.gestureControlTreeModelViewAndViewLayout.addWidget(self.gestureTreeInfo, 3)
        self.statusLayout.addWidget(self.statusOutput, 1)
        self.gestureData = []
        self.gestureControlTreeLabel = qtw.QLabel("Gesture Management")
        self.statusFrame.setLayout(self.statusLayout)
        self.modelMakerTabLayout.addWidget(self.statusFrame, 3, 0, 1, -1)
        self.outerGestureControlTreeBtnLayout.addWidget(self.gestureControlTreeLabel, 0)
        self.outerGestureControlTreeBtnLayout.addLayout(self.gestureControlTreeBtnLayout, 1)
        self.treeAndCameraLayout.addLayout(self.outerGestureControlTreeBtnLayout)
        self.modelMakerTabLayout.addLayout(self.cameraViewLayout, 1, 1, 1, -1)
        self.outerGestureControlTreeBtnLayout.addLayout(self.gestureControlTreeModelViewAndViewLayout, 2)
        self.modelMakerTabLayout.addLayout(self.treeAndCameraLayout, 1, 0)
        self.modelMakerTabLayout.addWidget(self.datasetPathLabel, 4, 0, 1, -1)
        self.modelMakerTab.setLayout(self.modelMakerTabLayout)
        self.statusOutput.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.gestureTreeInfo.header().setSectionResizeMode(qtw.QHeaderView.ResizeMode.Stretch)
        self.listGesturesTree.header().setStretchLastSection(True)
        self.gestureTreeInfo.header().setSectionResizeMode(qtw.QHeaderView.ResizeMode.Stretch)
        self.listGesturesTree.header().setStretchLastSection(True)
        return self.modelMakerTab

    def autocorrecting(self, word):
        """Find nearest known word candidate for a recognized token."""
        matches = get_close_matches(
            word,
            self.wordSet,
            n=1,
            cutoff=0.85
        )
        return matches[0] if matches else None
    
    def wordPercentage(self, word, maxLen=8):
        """Compute a simple confidence heuristic for word completeness."""
        if word not in self.wordSet:
            return 0.0
        lengthScore = min(len(word) / maxLen, 1.0)
        return 0.6 + 0.4 * lengthScore
    
    def checkWordBoundary(self):
        """Flush buffered ASL letters into finalized words after idle gap."""
        if self.decoder.shouldFlush():
            # tag values:
            # auto = autocorrected word, word = accepted raw word,
            # letters = fallback when confidence is too low.
            text, conf, tag = self.decoder.flush()
            self.aslTranscriptionOutput.append(
                f"{text} ({tag}, {conf:.2f})"
            )
            self.aslTranscriptionOutput.setText("…")
    
    def aslmodelshow(self):
        """Render MediaPipe hand landmarks on the current frame."""
        self.mpHands = mp.solutions.hands
        self.mpDrawing = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.8
        )
        self.frame = cv2.flip(self.frame, 1)
        self.rgbFrame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(self.rgbFrame)
        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                self.mpDrawing.draw_landmarks(self.frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        cv2.imshow('Hand Detection', self.frame)
    
    def stopCamera(self):
        """Stop frame timer and release camera handle if open."""
        self.frameTimer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def startCamera(self):
        """Start camera using configured/working index and resume frame timer."""
        if not self.cap or not self.cap.isOpened():
            preferred = SETTINGS.app.camera
            working = findWorkingCamera(preferred)
            if working is None:
                working = findWorkingCamera(0)
            if working is None:
                raise RuntimeError("No working camera detected.")
            if working != preferred:
                ConfigAPI.update("app", "camera", working)
            self.cap = cv2.VideoCapture(working)
        self.frameTimer.start(33)
    
    def onTabChanged(self, index):
        """Enable camera timer only on tabs that need a live feed."""
        self.widget = self.tabs.widget(index)
        # Camera is only needed on tabs that render live video.
        if self.widget in (self.translatorTab, self.modelMakerTab):
            if not self.frameTimer.isActive():
                self.frameTimer.start(33)
                self.logStatus("Camera on", LogLevel.INFO)
        else:
            self.frameTimer.stop()
            self.logStatus("Camera Stopped", LogLevel.INFO)

    def readWorkerLogs(self):
        """Read new worker log lines and forward them into UI loggers."""
        try:
            with open(WORKER_LOG_PATH, "r") as f:
                # Tail behavior: consume only newly appended lines each poll.
                lines = f.readlines()[self.lastWorkerLogLine:]
                self.lastWorkerLogLine += len(lines)
            for line in lines:
                level, msg = self.parseWorkerLogLine(line)
                self.workerLogger.log(msg, level)
                self.logStatus(f"Message From Model Training Software: {msg}", level)
        except FileNotFoundError:
            pass

    def parseWorkerLogLine(self, line):
        """Infer log level from worker line prefixes."""
        if "[ERROR]" in line:
            return LogLevel.ERROR, line.strip()
        if "[WARNING]" in line:
            return LogLevel.WARNING, line.strip()
        if "[DEBUG]" in line:
            return LogLevel.DEBUG, line.strip()
        return LogLevel.INFO, line.strip()
    
    def toggleAudioRecording(self):
        """Start/stop Whisper transcription worker and update button state."""
        # Worker thread acts like a toggle; wait() ensures clean stop before
        # allowing another start.
        if self.whisperWorker.isRunning():
            self.whisperWorker.stop()
            self.whisperWorker.wait()
            self.audioRecordBtn.setText("Start Audio Transcription")
        else:
            self.whisperWorker.running = True
            self.whisperWorker.start()
            self.audioRecordBtn.setText("Stop Audio Transcription")
        
    def updateTranscription(self, text):
        """Append timestamped microphone transcription text to UI."""
        # Append-only cursor update avoids clearing existing transcript history.
        self.ts = datetime.now().strftime("[%H:%M:%S] ")
        self.cursor = self.transcriptionOutput.textCursor()
        self.cursor.movePosition(qtg.QTextCursor.MoveOperation.End)
        self.cursor.insertText(self.ts + text.strip() + "\n")
        self.transcriptionOutput.setTextCursor(self.cursor)
        self.transcriptionOutput.ensureCursorVisible()

    @qtc.pyqtSlot(str, float)
    def updateASLTranscription(self, name, score):
        """Push recognized sign token into decoder and show live preview."""
        # Live preview is letter stream; final word decision happens by timer.
        preview = self.decoder.addLetter(name)
        self.aslTranscriptionOutput.setText(preview)     

    def scoreTranscription(self, score, name):
        """Append per-token confidence score in the score pane."""
        # Keep scoring stream separate from text stream so confidence can be
        # reviewed independently.
        self.ts = datetime.now().strftime("[%H:%M:%S] ")
        self.cursor = self.scoreTranscriptionOutput.textCursor()
        self.cursor.movePosition(qtg.QTextCursor.MoveOperation.End)
        self.cursor.insertText(f"{self.ts}{name} ({score:.2f})\n")
        self.scoreTranscriptionOutput.setTextCursor(self.cursor)
        self.scoreTranscriptionOutput.ensureCursorVisible()
        
    def gestureNameExistsCheck(self):
        """Validate gesture-name input before creating a new gesture entry."""
        if self.gestureNameInput.text().strip():
           self.addGesture(self.gestureNameInput.text().strip())
        else:
            self.errorMenu(message="The Gesture Does Not Have a Name.")

    def loadData(self):
        """Load gesture metadata JSON, normalizing legacy schema when needed."""
        # Legacy safety: normalize old dict schema into the current list schema.
        if not os.path.exists(JSON_FILE):
            return []
        with open(JSON_FILE, "r") as f:
            self.data = json.load(f)
        if isinstance(self.data, dict):
            self.logStatus("WARNING: gestures.json is not a list — fixing automatically", LogLevel.ERROR)
            self.data = [self.data]
        if not isinstance(self.data, list):
            self.logStatus("ERROR: gestures.json is invalid", LogLevel.ERROR)
            return []
        return self.data
    
    def saveData(self, data):
        """Persist gesture metadata list to JSON file."""
        with open(JSON_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def countImages(self, directory):
        """Count image files in a gesture folder."""
        if not os.path.exists(directory):
            return 0
        return sum(1 for file in os.listdir(directory) if file.lower().endswith(IMAGE_EXTENSIONS))
    
    def addGesture(self, name):
        """Create gesture folder and metadata entry if name is new."""
        self.data = self.loadData()
        if any(self.entry["name"] == name for self.entry in self.data):
            self.logStatus(f"Entry '{name}' already exists.")
            return
        self.imageDir = Path(DATASET_PATH) / name
        self.imageDir.mkdir(parents=True, exist_ok=True)
        self.imageCount = self.countImages(self.imageDir)
        self.data.append({
            "name": name,
            "image_count": self.imageCount
        })
        self.saveData(self.data)
        # Refresh trees immediately so users see new gesture/count.
        self.loadExistingGestures()
        self.gestureNameInput.clear()
        self.logStatus(f"Added '{name}' with {self.imageCount} images.")

    def updateAllImageCounts(self):
        """Recalculate and persist image counts for all gesture entries."""
        self.data = self.loadData()
        for self.entry in self.data:
            self.folder = os.path.join(DATASET_PATH, self.entry["name"])
            self.entry["image_count"] = self.countImages(self.folder)
        self.saveData(self.data)
        self.logStatus("Image counts updated.", LogLevel.INFO)
    
    def deleteGesture(self, name):
        """Delete gesture metadata/folder and reset related capture state."""
        # Pause rendering and camera thread before mutating gesture state/files.
        self.frameTimer.stop()
        self.listGesturesTree.setCurrentItem(None)
        self.gestureTreeInfo.setCurrentItem(None)
        self.currentGesture = None
        self.data = self.loadData()
        self.stopEvent.set()
        if self.cameraThread:
            self.cameraThread.join(timeout=0.2)
        # Remove metadata first, then remove dataset folder from disk.
        self.data = [d for d in self.data if d["name"] != name]
        self.saveData(self.data)
        self.listGesturesTree.blockSignals(True)
        self.gestureTreeInfo.blockSignals(True)
        self.gestureDir = Path(DATASET_PATH) / name
        if self.gestureDir.exists():
            shutil.rmtree(self.gestureDir)
        self.logStatus(f"Deleted gesture '{name}'", LogLevel.INFO)
        self.name = None
        self.currentGesture = None
        self.capturing = False
        # Resume UI loop after state and files are consistent again.
        self.frameTimer.start(33)
        self.loadExistingGestures()

    def loadExistingGestures(self, orderByName=True):
        """Rebuild gesture tree widgets from persisted metadata."""
        # Tree widgets are rebuilt from JSON each time to avoid stale counts.
        self.listGesturesTree.clear()
        self.gestureTreeInfo.clear()
        self.data = self.loadData()
        if orderByName:
            self.data = sorted(self.data, key=lambda x: x["name"].lower())
        self.gestures = self.data
        for self.entry in self.data:
            self.name = self.entry["name"]
            self.count = self.entry["image_count"]
            self.listGesturesTree.addTopLevelItem(
            qtw.QTreeWidgetItem([self.name])
            )
            self.gestureTreeInfo.addTopLevelItem(
                qtw.QTreeWidgetItem([self.name, str(self.count)])
            )
        return self.data
        
    def refreshGestures(self):
        """Reload gesture trees from disk-backed metadata."""
        self.loadExistingGestures(orderByName=True)
        self.logStatus("Refreshed Gesture Tree", LogLevel.INFO)

    def gestureSelectedCheck(self):
        """Guard delete flow by requiring a selected gesture first."""
        self.item = self.selectedGesture()
        if self.item:
            self.confirmGestureDelete(self.item)
        else:
            self.errorMenu(message="A Gesture is Not Selected.")

    def selectedGesture(self):
        """Return selected gesture item and sync selection in info tree."""
        # Keep both trees visually synchronized to the same selected gesture.
        self.item = self.listGesturesTree.currentItem()
        if not self.item:
            return None
        self.name = self.item.text(0)
        for i in range(self.gestureTreeInfo.topLevelItemCount()):
            self.infoItem = self.gestureTreeInfo.topLevelItem(i)
            if self.infoItem.text(0) == self.name:
                self.gestureTreeInfo.setCurrentItem(self.infoItem)
                break
        return self.item
    
    def confirmGestureDelete(self, item):
        """Ask for confirmation before deleting the selected gesture."""
        item = self.listGesturesTree.currentItem()
        if not item:
            return
        self.name = item.text(0)
        self.reply = qtw.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the gesture: '{self.name}'?",
            qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No,
            qtw.QMessageBox.StandardButton.No
        )
        if self.reply == qtw.QMessageBox.StandardButton.Yes:
            self.frameTimer.stop()
            # Force capture-off before delete to prevent writes to removed dir.
            if self.capturing:
                self.capturing = False
                self.startCaptureBtn.setText("Start Capture")
            self.deleteGesture(self.name)
            self.frameTimer.start(33)

    def startCapture(self):
        """Convenience handler that delegates to capture toggle logic."""
        self.toggleCapture()

    def initCamera(self):
        """Initialize camera capture object and shared thread primitives."""
        # Try a few device indexes and keep the first camera that can read frames.
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            self.cap = self.cap
        else:
            self.cap = None
            for i in range(4):
                self.tmp = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if self.tmp.isOpened():
                    self.ret, _ = self.tmp.read()
                    if self.ret:
                        self.cap = self.tmp
                        break
                    self.tmp.release()
        if not self.cap or not self.cap.isOpened():
            self.errorMenu(message="No camera found")
            self.cap = None
            return             
        self.cap = self.cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.logStatus(f"Found Camera {self.cap.isOpened()}", LogLevel.INFO)
        # Keep capture buffer small to minimize preview/inference lag.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        # Reader thread state shared with the UI refresh timer.
        self.stopEvent = threading.Event()
        self.frameLock = threading.Lock()
        self.frame = None
        self.cameraThread = None

    def cameraLoop(self):
        """Continuously read frames and publish only the newest frame."""
        # Dedicated capture thread: always keep latest frame in memory.
        while not getattr(self, "stopEvent", threading.Event()).is_set() and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Overwrite previous frame; downstream consumers only need latest.
                with self.frameLock:
                    self.frame = frame
    
    def launchCameraThread(self):
        """Start background camera reader thread if it is not already running."""
        if getattr(self, "cameraThread", None) and self.cameraThread.is_alive():
            return
        self.stopEvent.clear()
        self.cameraThread = threading.Thread(target=self.cameraLoop, daemon=True)
        self.cameraThread.start()
        self.logStatus("Camera started", LogLevel.INFO)

    def updateFrame(self):
        """Render current frame, run gesture inference, and save captures."""
        # Render latest frame into camera widgets and optionally save samples.
        if not hasattr(self, "frameLock"):
            return
        self.frameCopy = None
        with self.frameLock:
            if self.frame is None:
                return
            self.frameCopy = self.frame.copy()
        self.rgb = cv2.cvtColor(self.frameCopy, cv2.COLOR_BGR2RGB)
        self.h, self.w, self.ch = self.rgb.shape
        self.bytesPerLine = self.ch * self.w
        if self.currentGesture:
            cv2.putText(self.frameCopy, f"{self.currentGesture}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        rgb = cv2.cvtColor(self.frameCopy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = qtg.QImage(rgb.data, w, h, ch * w, qtg.QImage.Format.Format_RGB888)
        self.pixmap = qtg.QPixmap.fromImage(qimg)
        if self.cameraView.isVisible():
            self.cameraView.setPixmap(self.pixmap)
        if self.translatorCameraView.isVisible():
            # Inference consumes raw BGR frame copy; UI shows converted pixmap.
            self.frameForGesture.emit(self.frameCopy)
            self.translatorCameraView.setPixmap(self.pixmap)
        if self.capturing and self.currentGesture:
            # Timestamped filenames prevent collisions during high-speed capture.
            self.gesture = self.currentGesture
            self.gestureDir = self.modelDir / self.gesture
            self.gestureDir.mkdir(parents=True, exist_ok=True)
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            self.filename = self.gestureDir / f"{self.timestamp}.jpg"
            cv2.imwrite(str(self.filename), self.frameCopy)

    def toggleCapture(self):
        """Toggle dataset image capture for the currently selected gesture."""
        # Capture mode writes timestamped frames into the selected gesture folder.
        if not self.cap:
            self.errorMenu(message="No camera available.")
            return
        self.item = self.listGesturesTree.currentItem()
        if not self.item:
            self.errorMenu(message="No gesture selected")
            return
        self.sel = self.item.text(0)
        self.gesture = self.sel
        self.currentGesture = self.sel
        self.capturing = not self.capturing
        if self.capturing:
            self.startCaptureBtn.setText("Stop Capture")
            self.gestureDir = os.path.join(self.modelDir, self.gesture)
            os.makedirs(self.gestureDir, exist_ok=True)
            self.logStatus(f"Started Capture for gesture '{self.gesture}'.", LogLevel.INFO)
        else:
            # Recount on stop so UI reflects files that were just written.
            self.startCaptureBtn.setText("Start Capture")
            self.updateAllImageCounts()
            self.refreshGestures()
        self.state = "ON" if self.capturing else "OFF"
        self.logStatus(f"Capture {self.state} for '{self.gesture}' gesture", LogLevel.INFO)

    def visualizeModel(self):
        """Preview a fixed number of example images per gesture label."""
        # Build label list from dataset directories so preview mirrors train data.
        self.logStatus(DATASET_PATH, LogLevel.DEBUG)
        self.labels = []
        for i in os.listdir(DATASET_PATH):
            if os.path.isdir(os.path.join(DATASET_PATH, i)):
                self.labels.append(i)
        self.logStatus(self.labels, LogLevel.INFO)
        for self.label in self.labels:
            self.labelDir = os.path.join(DATASET_PATH, self.label)
            self.exampleFilenames = os.listdir(self.labelDir)[:NUM_EXAMPLES]
            # Show a horizontal sample strip per label for quick data sanity checks.
            self.fig, self.axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
            for i in range(NUM_EXAMPLES):
                self.axs[i].imshow(plt.imread(os.path.join(self.labelDir, self.exampleFilenames[i])))
                self.axs[i].get_xaxis().set_visible(False)
                self.axs[i].get_yaxis().set_visible(False)
            self.fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {self.label}')
        plt.show()
        return self.labels
    
    def runTraining(self):
        train_script = Path(__file__).parent.parent.parent / "app/scripts/train.py"
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=True,
            text=True
        )
        self.logStatus(f"Training exited with code {result.returncode}", LogLevel.INFO)
        if result.stdout:
            self.logStatus(f"Training output: {result.stdout}", LogLevel.INFO)
        if result.stderr:
            self.logStatus(f"Training errors: {result.stderr}", LogLevel.ERROR)

    def trainExportModel(self):
        """User-facing wrapper that announces and starts model training."""
        # Keep this method lightweight so UI button handler stays clear.
        self.logStatus("Making the model", LogLevel.INFO)
        self.logStatus("This may take anywhere from 30 Seconds to 20 Minutes depending on hardware capabilities and amount of images being trained.", LogLevel.INFO)
        self.runTraining()

    def toggleDebugLogging(self, state):
        """Switch logger verbosity between INFO and DEBUG."""
        if state == qtc.Qt.CheckState.Checked:
            self.runtimeLogger.setLevel(LogLevel.DEBUG)
            self.workerLogger.setLevel(LogLevel.DEBUG)
            self.translatorLogger.setLevel(LogLevel.DEBUG)
            self.logStatus("Debug logging enabled", LogLevel.INFO)
        else:
            self.runtimeLogger.setLevel(LogLevel.INFO)
            self.translatorLogger.setLevel(LogLevel.INFO)
            self.workerLogger.setLevel(LogLevel.INFO)
            self.logStatus("Debug logging disabled", LogLevel.INFO)
    
    def openVersionFolder(self):
        """Open dataset/model directory in file explorer."""
        os.startfile(self.modelDir)
    
    def settingsTabUI(self):
        """Construct settings tab and wire display/camera/config controls."""
        # Settings tab centralizes display, camera, and debug controls.
        self.settingsTab = qtw.QWidget()
        self.settingsTabLayout = qtw.QGridLayout()
        self.visualizeModelExamplesInput = qtw.QLineEdit()
        self.resetSettingsButton = qtw.QPushButton()
        self.resetSettingsButton.setText("Reset All Settings Back to Defaults")
        self.visualizeModelExamplesInput.setPlaceholderText("Change Number of Examples Shown When Visualizing the Model: ")
        self.debugCheckbox = qtw.QCheckBox("Enable debug logging")
        self.resolutionMenu = qtw.QComboBox()
        self.cameraMenu = qtw.QComboBox()
        self.windowModeMenu = qtw.QComboBox()
        self.monitorMenu = qtw.QComboBox()
        self.resolutionLayout = qtw.QVBoxLayout()
        self.monitorLayout = qtw.QVBoxLayout()
        self.cameraSettingsLayout = qtw.QVBoxLayout()
        self.setCameraLabel = qtw.QLabel("Set the Camera You Would Like To Use")
        self.resolutionLabel = qtw.QLabel("Change Window Size and Aspect Ratio")
        self.aspectRationScaleLabel = qtw.QLabel("Select Aspect Ratio Scale")
        self.windowResolutionLabel = qtw.QLabel("Select a Resolution")
        self.windowModeLabel = qtw.QLabel("Fullscreen?")
        self.dpiCheck = qtw.QCheckBox("Enable DPI Scaling")
        self.dpiCheck.setChecked(SETTINGS.app.dpi_scaling)
        self.cameraSettingsLayout.addWidget(self.setCameraLabel, 0)
        self.cameraSettingsLayout.addWidget(self.cameraMenu, 1)
        self.resolutionLayout.addWidget(self.dpiCheck)
        self.resolutionLayout.addWidget(self.resolutionLabel, 0)
        self.resolutionLayout.addWidget(self.windowModeLabel, 1)
        self.resolutionLayout.addWidget(self.windowModeMenu, 2)
        self.resolutionLayout.addWidget(self.aspectRationScaleLabel, 3)
        self.aspectRatioScaleMenu = qtw.QComboBox()
        self.resolutionLayout.addWidget(self.aspectRatioScaleMenu, 4)
        self.resolutionLayout.addWidget(self.windowResolutionLabel, 5)
        self.resolutionLayout.addWidget(self.resolutionMenu, 6)

        self.windowModeMenu.setCurrentText(self.windowManager.mode.title())
        
        for i, screen in enumerate(qtw.QApplication.screens()):
            self.monitorMenu.addItem(f"Monitor {i}: {screen.size().width()}x{screen.size().height()}", i)

        self.monitorMenu.currentIndexChanged.connect(self.changeMonitor)

        self.aspectRatioScaleMenu.addItems(["16:9", "4:3"])
        
        self.windowModeMenu.addItems([
            "Windowed",
            "Fullscreen",
            "Borderless Fullscreen"
        ])
        
        self.resolutions = {
            "16:9": [
                ("3840x2160", 3840, 2160),
                ("2560x1440", 2560, 1440),
                ("1920x1080", 1920, 1080),
                ("1280x720", 1280, 720),
                ("640x480", 640, 480)
            ],
            "4:3": [
                ("2048x1536", 2048, 1536),
                ("1920x1440", 1920, 1440),
                ("1400x1050", 1400, 1050),
                ("1280x960", 1280, 960),
                ("1024x768", 1024, 768),
                ("800x600", 800, 600),
                ("640x480", 640, 480)
            ]
        }

        self.monitorLayout.addWidget(qtw.QLabel("Monitor"), 1)
        self.monitorLayout.addWidget(self.monitorMenu, 1)

        self.widthInput = None
        self.heightInput = None

        self.windowModeMenu.currentTextChanged.connect(self.changeWindowMode)
        self.aspectRatioScaleMenu.currentTextChanged.connect(self.updateWindowResolutions)
        self.resolutionMenu.currentIndexChanged.connect(self.updateWindowSizeValues)

        self.dpiCheck.stateChanged.connect(self.toggleDPIScaling)
        self.cameraMenu.currentIndexChanged.connect(lambda: ConfigAPI.update("app","camera", self.cameraMenu.currentData()))


        # Prime the resolution menu using the active aspect ratio option.
        self.updateWindowResolutions(self.aspectRatioScaleMenu.currentText())

        self.settingsTabLayout.addLayout(self.monitorLayout, 0, 0)
        self.settingsTabLayout.addLayout(self.cameraSettingsLayout, 1, 0)
        self.settingsTabLayout.addLayout(self.resolutionLayout, 2, 0)
        self.settingsTabLayout.addWidget(self.debugCheckbox, 3, 0)
        self.settingsTabLayout.addWidget(self.visualizeModelExamplesInput, 4, 0)
        self.settingsTabLayout.addWidget(self.resetSettingsButton, 5, 1)

        self.debugCheckbox.stateChanged.connect(self.toggleDebugLogging)
        self.resetSettingsButton.pressed.connect(self.confirmResetSettings)
        self.settingsTab.setLayout(self.settingsTabLayout)
        self.monitorMenu.setCurrentIndex(self.windowManager.monitorIndex)
        #return self.settingsTab
    
    def changeMonitor(self,i):
        """Apply monitor change and refresh valid resolution options."""
        # Resolution list depends on active monitor dimensions.
        self.windowManager.apply(monitor=i)
        self.updateWindowResolutions(self.aspectRatioScaleMenu.currentText())
    
    def toggleDPIScaling(self, state):
        """Toggle DPI scaling flag and apply policy immediately."""
        # Persisted through WindowManager and applied immediately.
        self.windowManager.dpiScaling = bool(state)
        self.windowManager.applyDPI()
    
    def updateWindowSizeValues(self):
        """Apply selected window resolution while in windowed mode."""
        # Ignore size updates while fullscreen/borderless is active.
        if self.windowManager.mode != WindowMode.WINDOWED:
            return
        data = self.resolutionMenu.currentData()
        if data:
            w,h = data
            self.widthInput = w
            self.heightInput = h
            self.windowManager.apply(width=w, height=h)
    
    def changeWindowMode(self, text):
        """Map settings UI label to internal window mode and apply it."""
        modeMap = {
            "Windowed": WindowMode.WINDOWED,
            "Fullscreen": WindowMode.FULLSCREEN,
            "Borderless Fullscreen": WindowMode.BORDERLESS
        }
        self.windowManager.apply(mode=modeMap[text])
    
    def updateWindowResolutions(self, aspect):
        """Populate resolutions filtered by selected aspect ratio."""
        self.resolutionMenu.clear()
        for w,h in self.windowManager.availableResolutions():
            # Keep aspect choices strict so listed options are predictable.
            if aspect=="16:9" and abs((w/h)-(16/9))>0:
                continue
            if aspect=="4:3" and abs((w/h)-(4/3))>0:
                continue
            self.resolutionMenu.addItem(f"{w}x{h}", (w,h))
            
    def keyPressEvent(self, e):
        """Exit fullscreen/borderless back to windowed on Escape key."""
        if e.key()==qtc.Qt.Key.Key_Escape and self.windowManager.mode!=WindowMode.WINDOWED:
            self.windowManager.apply(mode=WindowMode.WINDOWED)

    def listAvailableCameras(self, max_tested=10):
        """Return list of camera indexes that open successfully."""
        cams = []
        # Enumerate quickly at startup to populate camera selection menu.
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(i)
                cap.release()
        return cams
    
    def updateSettings(self):
        """Collect settings form values and persist them to config file."""
        # Read current UI values and persist them back to config.
        # App settings.
        self.newFullscreenMode = self.windowManager.mode
        self.newWidth = self.widthInput
        self.newHeight = (self.heightInput)
        self.setLogLevel = (self.logLevelInput)
        # Gesture settings.
        self.newGesturesName = self.gestureModelInput
        # Control/transcription settings.
        self.newExampleAmount = int(self.visualizeModelExamplesInput.text())
        self.newSampleRate = self.sampleRateInput
        self.newInitialChunkDeration = self.initialChunkDerationInput
        self.newMinimumChunkDeration = self.minimumChunkDerationInput
        self.newChunkDecrement = self.chunkDecrementInput
        self.linesBool = bool(self.linesCheckBoxInput)
        self.newConfidenceThreshold = self.confidenceThresholdInput
        self.setAutocorrectToggle = bool(self.AutocorrectToggleInput)
        self.setAutocorrectThreshold = self.AutocorrectThresholdInput
        self.setWordGap = self.setWordGapInput
        self.setPreviewToggle = bool(self.PreviewToggleInput)
        self.setConfidenceToggle = bool(self.ConfidenceToggleInput)
        # Writes are intentionally explicit per key to keep config errors local.
        # Write app settings.
        ConfigAPI.update("app", "fullscreen_mode", self.newFullscreenMode)
        ConfigAPI.update("app", "width", self.newWidth)
        ConfigAPI.update("app", "height", self.newHeight)
        ConfigAPI.update("app", "log_level", self.setLogLevel)
        # Write gesture settings.
        ConfigAPI.update("gestures", "gesture_model", self.newGesturesName)
        # Write control settings.
        ConfigAPI.update("settings", "examples", self.newExampleAmount)
        ConfigAPI.update("settings", "sam_rate", self.newSampleRate)
        ConfigAPI.update("settings", "init_chunk_der", self.newInitialChunkDeration)
        ConfigAPI.update("settings", "min_chunk_der", self.newMinimumChunkDeration)
        ConfigAPI.update("settings", "chunk_dec", self.newChunkDecrement)
        ConfigAPI.update("settings", "lines", self.linesBool)
        ConfigAPI.update("settings", "confidence_threshold", self.newConfidenceThreshold)
        ConfigAPI.update("settings", "autocorrect", self.setAutocorrectToggle)
        ConfigAPI.update("settings", "autocorrect_threshold", self.setAutocorrectThreshold)
        ConfigAPI.update("settings", "word_gap", self.setWordGap)
        ConfigAPI.update("settings", "preview_toggle", self.setPreviewToggle)
        ConfigAPI.update("settings", "confidence_toggle", self.setConfidenceToggle)

    # Don't really know if this is needed as I am pretty sure it calls the settings from the file everytime it is needed.
    def reloadSettings(self):
        """Placeholder hook for settings reload behavior."""
        SETTINGS

    def confirmResetSettings(self):
        """Prompt user before resetting persisted settings to defaults."""
        self.reply = qtw.QMessageBox.question(
            self,
            "Confirm Reset",
            f"Are you sure you want to reset all of the settings back to defaults?",
            qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No,
            qtw.QMessageBox.StandardButton.No
        )
        if self.reply == qtw.QMessageBox.StandardButton.Yes:
            loadDefaultSettings()
    
    def errorMenu(self, message):
        """Show a blocking error dialog with the provided message."""
        qtw.QMessageBox.critical(self, "Error: ", message, qtw.QMessageBox.StandardButton.Ok)
    
    def logStatus(self, message, level=LogLevel.INFO):
        """Broadcast one log message to all UI log panes."""
        self.runtimeLogger.log(message, level)
        self.translatorLogger.log(message, level)
        self.workerLogger.log(message, level)

    def closeEvent(self, event):
        """Stop workers, release resources, persist window state, then close."""
        # Gracefully stop threads/devices before writing final window state.
        if hasattr(self, "stopEvent"):
            self.stopEvent.set()
        if hasattr(self, "cameraThread") and self.cameraThread:
            self.cameraThread.join(timeout=1)
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        if hasattr(self, "whisperWorker"):
            self.whisperWorker.stop()
            self.whisperWorker.wait()
        if hasattr(self, "gestureThread"):
            self.gestureThread.quit()
            self.gestureThread.wait()
        qtw.QApplication.quit()
        self.windowManager.saveState()
        # Persist final geometry/mode so next startup restores the same layout.
        ConfigAPI.update("app","width", self.windowManager.width)
        ConfigAPI.update("app","height", self.windowManager.height)
        ConfigAPI.update("app","pos_x", self.windowManager.posx)
        ConfigAPI.update("app","pos_y", self.windowManager.posy)
        ConfigAPI.update("app","fullscreen_mode", self.windowManager.mode)
        ConfigAPI.update("app","monitor", self.windowManager.monitorIndex)
        super().closeEvent(event)


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec())
