#!/usr/bin/env python3
from config.loader import load_settings
#
#
# Fuck you script
#
#
#
# GOOD FUCKING LUCK WITH main.py :3 :)
# you will not have fun 
# if it works DO NOT FUCKING TOUCH IT
#
#
#from mediapipe_model_maker import gesture_recognizer as mp
#assert tf.__version__.startswith('2')
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
#from PIL import Image, ImageTk
from datetime import datetime
import PyQt6.QtWidgets as qtw
#import ttkbootstrap as tb
import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg
from pathlib import Path
#import tensorflow as tf
#import mediapipe as mp
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import soundcard as sc
import soundfile as sf
import numpy as np
import io
import mysql.connector
import tomli
import threading
import requests
import subprocess
import json
import sqlite3
import shutil
import time
import sys
import cv2
import re
import os 
SETTINGS = load_settings()
HF_TOKEN = SETTINGS.env.hf_token
SHARED = Path(__file__).parent.parent.parent / "shared"
DB_FILE = Path(__file__).parent / "gestures.db"
DATASET_PATH = Path(__file__).parent.parent.parent / "shared/dataset"
EXPORT_PATH = Path(__file__).parent.parent.parent / "shared/exports"
WORKER_LOG_PATH = Path(__file__).parent.parent.parent / "shared/logs/worker.log"
MODEL_PATH = Path(__file__).parent.parent.parent / "deploy/gestures.task"
WORDLIST = Path(__file__).parent.parent.parent / "deploy/words.txt"
#EXPORT_PATH.mkdir(parents=True, exist_ok=True)
#DATASET_PATH.mkdir(parents=True, exist_ok=True)
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

class LogLevel:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

class UILogger(qtc.QObject):
    logReady = qtc.pyqtSignal(str)
    def __init__(self, name="app"):
        super().__init__()
        self.name = name
        self.level = 1

    def setLevel(self, level):
        self.level = level

    def log(self, message, level=1):
        if level < self.level:
            return
        ts = datetime.now().strftime("[%H:%M:%S]")
        self.logReady.emit(f"{ts} {message}")

class LogViewer(qtw.QTextEdit):
    def __init__(self, maxLines=500, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.maxLines = maxLines
        self.lines = []
        self.flushTimer = qtc.QTimer(self)
        self.flushTimer.timeout.connect(self.flush)
        self.flushTimer.start(100)  # 10 FPS UI updates
        self._pending = []

    def enqueue(self, message):
        self._pending.append(message)

    def flush(self):
        if not self._pending:
            return
        self.lines.extend(self._pending)
        self._pending.clear()
        if len(self.lines) > self.maxLines:
            self.lines = self.lines[-self.maxLines:]
        self.setPlainText("\n".join(self.lines))
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class WhisperWorker(qtc.QThread):
    textReady = qtc.pyqtSignal(str)
    logMessage = qtc.pyqtSignal(str, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChunkDuration = INITIAL_CHUNK_DURATION
        self.lastText = ""
        self.running = True
        self.mic = sc.default_microphone()
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        try:
            Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        except Exception as e:
            self.logMessage.emit(f"Diarization disabled: {e}", LogLevel.WARNING)
            self.diarization_pipeline = None

    def stop(self):
        self.running = False

    def transcribeAudio(self, audio):
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
        buffer.seek(0)
        segments, _ = self.model.transcribe(buffer, beam_size=5)
        return "\n".join(seg.text for seg in segments)

    def run(self):
        self.logMessage.emit("Loading Whisper model...", LogLevel.INFO)
        recorded = np.zeros((0, 1), dtype=np.float32)
        while self.running:
            with self.mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
                chunk = recorder.record(numframes=int(self.currentChunkDuration * SAMPLE_RATE))
            recorded = np.concatenate([recorded, chunk], axis=0)
            maxSamples = SAMPLE_RATE * 30
            recorded = recorded[-maxSamples:]
            text = self.transcribeAudio(recorded)
            #self.textReady.emit(text)
            if self.currentChunkDuration > MIN_CHUNK_DURATION:
                self.currentChunkDuration = max(self.currentChunkDuration - CHUNK_DECREMENT, MIN_CHUNK_DURATION)
            if text != self.lastText:
                newText = text[len(self.lastText):].strip()
                if newText:
                    self.textReady.emit(newText)
                self.lastText = text

class AspectRatioWidget(qtw.QWidget):
    def __init__(self, ratio=16/9, parent=None):
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
        self.pixmap = pixmap
        if pixmap is not None:
            self.label.setPixmap(pixmap.scaled(
                self.label.size(),
                qtc.Qt.AspectRatioMode.KeepAspectRatio,
                qtc.Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event):
        if self.pixmap:
            self.label.setPixmap(self.pixmap.scaled(
                self.label.size(),
                qtc.Qt.AspectRatioMode.KeepAspectRatio,
                qtc.Qt.TransformationMode.SmoothTransformation
            ))

    def paintEvent(self, event):
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
        self.painter.drawPixmap(
            qtc.QRect(self.x, self.y, self.target_w, self.target_h),
            self.pixmap
        )
        self.painter.end()

class GestureRecognizerWithoutLinesWorker(qtc.QObject):
    #frameReady = qtc.pyqtSignal(qtg.QPixmap)
    gestureRecognized = qtc.pyqtSignal(str, float)
    def __init__(self, MODEL_PATH: str, parent=None):
        super().__init__(parent)
        self.modelPath = str(MODEL_PATH)
        self.running = False
        self.enabled = True
        self.lastProcessTime = 0.0
        self.minInterval = 0.10
        self.baseOptions = python.BaseOptions(model_asset_path=str(self.modelPath))
        self.options = vision.GestureRecognizerOptions(base_options=self.baseOptions, running_mode = vision.RunningMode.VIDEO)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
        self.lastGesture = None
        
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
        frame = cv2.flip(frame, 1)
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
        timestamp = int(time.time() * 1000)
        self.result = self.recognizer.recognize_for_video(mpImage, timestamp)
        if self.result.gestures:
            top = self.result.gestures[0][0]
            if top.score > 0.5:
                if top.category_name != self.lastGesture:
                    self.lastGesture = top.category_name
                    self.gestureRecognized.emit(top.category_name, top.score)
            #self.gestureName = top.category_name
            #self.score = top.score
            #if self.result.gestures and self.topGesture.score > 0.5:
#                cv2.putText(frame, f"{self.gestureName} ({self.score:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #self.h, self.w, self.ch = frame.shape
        #self.bytesPerLine = self.ch * self.w
        #self.qimg = qtg.QImage(frame.data, self.w, self.h, self.bytesPerLine, qtg.QImage.Format.Format_BGR888)
        #self.pixmap = qtg.QPixmap.fromImage(self.qimg)
        #self.frameReady.emit(self.pixmap)

class MainGui(qtw.QMainWindow):
    frameForGesture = qtc.pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.title = SETTINGS.app.name
        self.width = SETTINGS.app.width
        self.height = SETTINGS.app.height
        self.setWindowTitle(self.title)
        self.runtimeLogger = UILogger("runtime")
        self.workerLogger = UILogger("worker")
        self.logStatus(f"Window size: {self.width}x{self.height}", LogLevel.DEBUG)
        self.resize(self.width, self.height)
        if self.width <= 0 or self.height <= 0:
            self.resize(1280, 800)
        self.logLevel = LogLevel.INFO
        self.modelDir = Path(DATASET_PATH)
        self.exportDir = Path(EXPORT_PATH)
        self.datasetPath = Path(DATASET_PATH)
        self.exportPath = Path(EXPORT_PATH)
        self.exampleAmount = NUM_EXAMPLES
        self.workerLogPath = Path(WORKER_LOG_PATH)
        self.lines = SETTINGS.settings.lines
        self.gestureThread = qtc.QThread(self)
        self.signRecognizerNoLines = GestureRecognizerWithoutLinesWorker(MODEL_PATH)
        self.signRecognizerNoLines.moveToThread(self.gestureThread)
        self.gestureThread.finished.connect(self.signRecognizerNoLines.deleteLater)
        self.gestureThread.start()
        self._logFilePos = 0
        self.workerLogTimer = qtc.QTimer()
        self.workerLogTimer.timeout.connect(self.readWorkerLogs)
        self.workerLogTimer.start(250)
        self.cameraViewLayout = qtw.QVBoxLayout()
        self.translatorCameraViewLayout = qtw.QVBoxLayout()
        self.statusLayout = qtw.QVBoxLayout()
        self.translatorStatusLayout = qtw.QVBoxLayout()
        self.capturing = False
        self.currentGesture = None
        os.makedirs(self.modelDir, exist_ok=True)
        self.workerLogPath = Path(__file__).parent.parent.parent / "shared/logs/worker.log"
        self._logFilePos = 0
        self.workerLogTimer = qtc.QTimer()
        #self.workerLogTimer.timeout.connect(self.readWorkerLogs)
        self.workerLogTimer.start(250)
        self.translatorCameraView = AspectRatioWidget(16/9)
        self.translatorCameraViewLayout.addWidget(self.translatorCameraView)
        self.cameraView = AspectRatioWidget(16/9)
        self.cameraViewLayout.addWidget(self.cameraView)
        self.outLayout = qtw.QVBoxLayout()
        self.translatorCameraView.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.cameraView.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.centralWid = qtw.QWidget()
        self.centralWid.setLayout(self.outLayout)
        self.setCentralWidget(self.centralWid)
        self.tabs = qtw.QTabWidget()
        self.quitTab = qtw.QWidget()
        #self.translatorStatusOutput = qtw.QTextEdit()
        #self.translatorStatusOutput.setReadOnly(True)
        #self.statusOutput = qtw.QTextEdit()
        #self.statusOutput.setReadOnly(True)
        #self.stdoutRedirector2 = TranslatorTextRedirector(self.translatorStatusOutput)
        #self.stderrRedirector2 = TranslatorTextRedirector(self.translatorStatusOutput)
        #self.stdoutRedirector = TextRedirector(self.statusOutput)
        #self.stderrRedirector = TextRedirector(self.statusOutput)
        #sys.stdout = self.stdoutRedirector and self.stdoutRedirector2
        #sys.stderr = self.stderrRedirector and self.stderrRedirector2
        #self.flushTimer.start(100)
        self.statusOutput = LogViewer(maxLines=400)
        self.translatorStatusOutput = LogViewer(maxLines=400)
        self.runtimeLogger.logReady.connect(self.statusOutput.enqueue)
        self.workerLogger.logReady.connect(self.translatorStatusOutput.enqueue)
        #print(self.stdoutRedirector)
        #print(self.stdoutRedirector2)
        #print(self.translatorStatusOutput)
        #print(self.statusOutput)
        #sys.stdout = self.stdoutRedirector2
        #sys.stderr = self.stderrRedirector2
        self.frame = None
        self.tabs.addTab(self.translatorTabUI(), "Translator")
        self.tabs.addTab(self.modelMakerTabUI(), "Model Maker")
        self.tabs.addTab(self.settingsTabUI(), "Settings")
        self.tabs.currentChanged.connect(lambda _: self.updateFrame())
        #self.tabs.currentChanged.connect(self.onTabChanged)
        self.outLayout.addWidget(self.tabs, 0)
        self.initCamera()
        self.gestures = self.loadExistingGestures(orderByName=True)
        self.loadExistingGestures(orderByName=True)
        self.frameTimer = qtc.QTimer()
        self.frameTimer.timeout.connect(self.updateFrame)
        self.frameTimer.start(33)
        self.whisperWorker = WhisperWorker()
        self.scoreTranscriptionOutput = qtw.QTextEdit()
        self.scoreTranscriptionOutput.setReadOnly(True)
        self.wordTimer = qtc.QTimer()
        self.wordTimer.timeout.connect(self.checkWordBoundary)
        self.wordTimer.start(200)
        #self.signRecognizerNoLines.frameReady.connect(self.translatorCameraView.setPixmap)
        self.whisperWorker.textReady.connect(self.updateTranscription)
        self.frameForGesture.connect(self.signRecognizerNoLines.processFrame)
        self.letterBuffer = []
        self.lastGestureTime = None
        self.gapBetweenSignRecognition = 1.0
        self.gestureInterval = 0.12
        self._lastGestureTime = 0.0
        self.signRecognizerNoLines.gestureRecognized.connect(self.updateASLTranscription)
        self.wordSet = set()
        self.whisperWorker.logMessage.connect(self.logStatus)
        if not WORDLIST.exists():
            raise FileNotFoundError(f"Word list not found: {WORDLIST}")
        with WORDLIST.open("r", encoding="utf-8") as f:
            self.wordSet = {line.strip().upper() for line in f if line.strip()}
        if self.cap:
            self.launchCameraThread()
            self.updateFrame()

    def translatorTabUI(self):
        self.translatorTab = qtw.QWidget()
        self.translatorTabLayout = qtw.QGridLayout()
        #self.translatorTabLayout.addWidget(self.translatorStatusOutput, 1, 0)
        self.translatorTabLayout.addLayout(self.translatorCameraViewLayout, 0, 0)
        self.translatorTab.setLayout(self.translatorTabLayout)
        self.transcriptionOutput = qtw.QTextEdit()
        self.transcriptionOutput.setReadOnly(True)
        self.aslTranscriptionOutput = qtw.QTextEdit()
        self.aslTranscriptionOutput.setReadOnly(True)
        self.translatorTabLayout.addWidget(self.transcriptionOutput, 0, 2)
        self.translatorTabLayout.addWidget(self.aslTranscriptionOutput, 1, 2)
        #self.translatorTabLayout.addWidget(self.scoresTranscriptionScoresOutput, 2, 2)
        self.audioRecordBtn = qtw.QPushButton("Record Audio")
        self.audioRecordBtnStatusLabel = qtw.QLabel
        self.translatorTabLayout.addWidget(self.audioRecordBtn, 0, 1)
        self.translatorStatusFrame = qtw.QWidget()
        self.translatorStatusLayout.addWidget(self.translatorStatusOutput)
        self.translatorStatusFrame.setLayout(self.translatorStatusLayout)
        self.translatorTabLayout.addWidget(self.translatorStatusFrame, 1, 0)
        self.datasetPathLabel = qtw.QLabel(f"Log Path: {WORKER_LOG_PATH}\n"
                                           f"Current Model Loaded Model: {MODEL_PATH}")
        self.datasetPathLabel.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.datasetPathLabel.setStyleSheet("color: gray; font-size: 11px;")
        self.datasetPathLabel.setToolTip("This is where gesture image data is stored")
        self.datasetPathLabel.mousePressEvent = lambda e: os.startfile(DATASET_PATH)
        self.datasetPathLabel.setCursor(qtc.Qt.CursorShape.PointingHandCursor)
        self.translatorTabLayout.addWidget(self.datasetPathLabel, 4, 0, 1, -1)
        self.audioRecordBtn.setCheckable(True)
        self.audioRecordBtn.clicked.connect(self.toggleAudioRecording)
        return self.translatorTab
    
    def modelMakerTabUI(self):
        #
        # Layouts: 
        # modelMakerTabLayout() is the main layout for the tab so it's just the outer ring for the tab 
        # outerGestureControlTreeBtnLayout() is the layout for the gesture management section of this tab organizes the whole gesture management section
        # gestureControlTreeBtnLayout() is the layout for the buttons that control the gesture management just used for convienece for organization
        # gestureControlTreeModelViewAndViewLayout() is the layout for showing what gesture exist and the information tied to them
        #
        #
        # I stopped tracking it good fucking luck
        # It's fucked
        #
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
        #self.statusLayout.addWidget(self.workerLogger)
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
        self.quitProgramBtn.clicked.connect(self.closeProgram)
        self.datasetPathLabel = qtw.QLabel(f"Image Storage Path: {DATASET_PATH}\n"
                                           f"Current Model: {self.modelDir}")
        self.datasetPathLabel.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.datasetPathLabel.setStyleSheet("color: gray; font-size: 11px;")
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
        #self.listGesturesTree.itemClicked.connect(self.whichGestureSelected)
        self.gestureData = []
        self.gestureControlTreeLabel = qtw.QLabel("Gesture Management")
        self.statusFrame.setLayout(self.statusLayout)
        self.modelMakerTabLayout.addWidget(self.statusFrame, 3, 0, 1, -1)
        self.outerGestureControlTreeBtnLayout.addWidget(self.gestureControlTreeLabel, 0)
        self.outerGestureControlTreeBtnLayout.addLayout(self.gestureControlTreeBtnLayout, 1)
        self.treeAndCameraLayout.addLayout(self.outerGestureControlTreeBtnLayout)
        self.modelMakerTabLayout.addLayout(self.cameraViewLayout, 1, 1, 1, -1)
        #self.treeAndCameraLayout.addLayout(self.cameraViewLayout, 2)
        #self.treeAndCameraLayout.addLayout(self.cameraViewLayout)
        self.outerGestureControlTreeBtnLayout.addLayout(self.gestureControlTreeModelViewAndViewLayout, 2)
        self.modelMakerTabLayout.addLayout(self.treeAndCameraLayout, 1, 0)
        self.modelMakerTabLayout.addWidget(self.datasetPathLabel, 4, 0, 1, -1)
        self.modelMakerTab.setLayout(self.modelMakerTabLayout)
        #self.cameraView.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        #self.listGesturesTree.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        #self.gestureTreeInfo.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.statusOutput.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        #self.treeAndCameraLayout.setStretchFactor(1, -1)
        #self.cameraViewLabel.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        #self.modelMakerTabLayout.setRowStretch(0, 0)
        #self.modelMakerTabLayout.setRowStretch(1, 1)
        #self.modelMakerTabLayout.setRowStretch(3, 0)
        #self.modelMakerTabLayout.setRowStretch(4, 0)
        #self.modelMakerTabLayout.setColumnStretch(0, 1)
        #self.treeAndCameraLayout.setStretch(0, 1)
        #self.treeAndCameraLayout.setStretch(1, 2)
        self.gestureTreeInfo.header().setSectionResizeMode(qtw.QHeaderView.ResizeMode.Stretch)
        self.listGesturesTree.header().setStretchLastSection(True)
        #self.statusFrame.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.gestureTreeInfo.header().setSectionResizeMode(qtw.QHeaderView.ResizeMode.Stretch)
        self.listGesturesTree.header().setStretchLastSection(True)
        return self.modelMakerTab
    
    def checkWordBoundary(self):
        if not self.letterBuffer or not self.lastGestureTime:
            return
        if time.time() - self.lastGestureTime >= self.gapBetweenSignRecognition:
            self.flushLetterBuffer()
    
    def aslmodelshow(self):
        self.mpHands = mp.solutions.hands
        self.mpDrawing = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.frame = cv2.flip(self.frame, 1)
        self.rgbFrame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(self.rgbFrame)
        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                self.mpDrawing.draw_landmarks(self.frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        cv2.imshow('Hand Detection', self.frame)
    
    def stopCamera(self):
        self.frameTimer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def startCamera(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.frameTimer.start(33)
    
    def onTabChanged(self, index):
        self.widget = self.tabs.widget(index)
        if self.widget in (self.translatorTab, self.modelMakerTab):
            if not self.frameTimer.isActive():
                self.frameTimer.start(33)
                self.logStatus("Camera on", LogLevel.INFO)
        else:
            self.frameTimer.stop()
            self.logStatus("Camera Stopped", LogLevel.INFO)

    def readWorkerLogs(self):
        try:
            with open(WORKER_LOG_PATH, "r") as f:
                lines = f.readlines()[self.lastWorkerLogLine:]
                self.lastWorkerLogLine += len(lines)
            for line in lines:
                level, msg = self.parseWorkerLogLine(line)
                self.workerLogger.log(msg, level)
        except FileNotFoundError:
            pass

    def parseWorkerLogLine(self, line):
        if "[ERROR]" in line:
            return LogLevel.ERROR, line.strip()
        if "[WARNING]" in line:
            return LogLevel.WARNING, line.strip()
        if "[DEBUG]" in line:
            return LogLevel.DEBUG, line.strip()
        return LogLevel.INFO, line.strip()
    
    def toggleAudioRecording(self):
        if self.whisperWorker.isRunning():
            self.whisperWorker.stop()
            self.whisperWorker.wait()
            self.audioRecordBtn.setText("Start Audio Transcription")
        else:
            self.whisperWorker.running = True
            self.whisperWorker.start()
            self.audioRecordBtn.setText("Stop Audio Transcription")
        
    def updateTranscription(self, text):
        self.ts = datetime.now().strftime("[%H:%M:%S] ")
        self.cursor = self.transcriptionOutput.textCursor()
        self.cursor.movePosition(qtg.QTextCursor.MoveOperation.End)
        self.cursor.insertText(self.ts + text.strip() + "\n")
        self.transcriptionOutput.setTextCursor(self.cursor)
        self.transcriptionOutput.ensureCursorVisible()
    

    def flushLetterBuffer(self):
        self.word = "".join(self.letterBuffer)
        if self.word in self.wordSet:
            self.output = self.word
        else:
            self.output = " ".join(self.letterBuffer)
        self.ts = datetime.now().strftime("[%H:%M:%S] ")
        self.cursor = self.aslTranscriptionOutput.textCursor()
        self.cursor.movePosition(qtg.QTextCursor.MoveOperation.End)
        self.cursor.insertText(self.ts + self.output + "\n")
        self.aslTranscriptionOutput.setTextCursor(self.cursor)
        self.aslTranscriptionOutput.ensureCursorVisible()
        self.letterBuffer.clear()
        self.lastGestureTime = None

    @qtc.pyqtSlot(str, float)
    def updateASLTranscription(self, name, score):
        self.currentGesture = name
        self.lastGestureTime = time.time()
        self.letterBuffer.append(name)
        
    def scoreTrascription(self, score, name):
        self.ts = datetime.now().strftime("[%H:%M:%S] ")
        self.cursor = self.scoreTranscriptionOutput.textCursor()
        self.cursor.movePosition(qtg.QTextCursor.MoveOperation.End)
        self.cursor.insertText(f"{self.ts}{name} ({score:.2f})\n")
        self.scoreTranscriptionOutput.setTextCursor(self.cursor)
        self.scoreTranscriptionOutput.ensureCursorVisible()
        
    def gestureNameExistsCheck(self):
        if self.gestureNameInput.text().strip():
           self.addGesture(self.gestureNameInput.text().strip())
        else:
            self.errorMenu(message="The Gesture Does Not Have a Name.")
    def loadData(self):
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
        with open(JSON_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def countImages(self, directory):
        if not os.path.exists(directory):
            return 0
        return sum(1 for file in os.listdir(directory) if file.lower().endswith(IMAGE_EXTENSIONS))
    
    def addGesture(self, name):
        self.data = self.loadData()
        if any(self.entry["name"] == name for self.entry in self.data):
            self.logStatust(f"Entry '{name}' already exists.")
            return
        self.imageDir = Path(DATASET_PATH) / name
        self.imageDir.mkdir(parents=True, exist_ok=True)
        self.imageCount = self.countImages(self.imageDir)
        self.data.append({
            "name": name,
            "image_count": self.imageCount
        })
        self.saveData(self.data)
        self.loadExistingGestures()
        self.gestureNameInput.clear()
        self.logStatus(f"Added '{name}' with {self.imageCount} images.")

    def updateAllImageCounts(self):
        self.data = self.loadData()
        for self.entry in self.data:
            self.folder = os.path.join(DATASET_PATH, self.entry["name"])
            self.entry["image_count"] = self.countImages(self.folder)
        self.saveData(self.data)
        self.logStatus("Image counts updated.", LogLevel.INFO)
    
    def deleteGesture(self, name):
        self.frameTimer.stop()
        self.listGesturesTree.setCurrentItem(None)
        self.gestureTreeInfo.setCurrentItem(None)
        self.currentGesture = None
        self.data = self.loadData()
        self.stopEvent.set()
        if self.cameraThread:
            self.cameraThread.join(timeout=0.2)
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
        self.frameTimer.start(33)
        self.loadExistingGestures()

    def loadExistingGestures(self, orderByName=True):
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
        self.loadExistingGestures(orderByName=True)
        self.logStatus("Refreshed Gesture Tree", LogLevel.INFO)

    def gestureSelectedCheck(self):
        self.item = self.selectedGesture()
        if self.item:
            self.confirmGestureDelete(self.item)
        else:
            self.errorMenu(message="A Gesture is Not Selected.")

    def selectedGesture(self):
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
            if self.capturing:
                self.capturing = False
                self.startCaptureBtn.setText("Start Capture")
            self.deleteGesture(self.name)
            self.frameTimer.start(33)
    def startCapture(self):
        self.toggleCapture()
    def initCamera(self):
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.logStatus(f"Found Camera {self.cap.isOpened()}", LogLevel.INFO)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.stopEvent = threading.Event()
        self.frameLock = threading.Lock()
        self.frame = None
        self.cameraThread = None

    def cameraLoop(self):
        while not getattr(self, "stopEvent", threading.Event()).is_set() and self.cap:
            self.ret, self.frame = self.cap.read()
            if self.ret:
                with self.frameLock:
                    self.frame = self.frame
    def launchCameraThread(self):
        if getattr(self, "cameraThread", None) and self.cameraThread.is_alive():
            return
        self.stopEvent.clear()
        self.cameraThread = threading.Thread(target=self.cameraLoop, daemon=True)
        self.cameraThread.start()
        self.logStatus("Camera started", LogLevel.INFO)

    def updateFrame(self):
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
        #if self.translatorCameraView.isVisible():
                #self.signRecognizerNoLines.processFrame(self.frameCopy)
        #if self.translatorCameraView.isVisible() and not self.lines:
        #self.now = time.time()
        #if self.now - self._lastGestureTime >= self.gestureInterval:
            #self.frameForGesture.emit(self.frameCopy)
            #self._lastGestureTime = self.now
        if self.cameraView.isVisible():
            self.cameraView.setPixmap(self.pixmap)
        if self.translatorCameraView.isVisible():
            self.frameForGesture.emit(self.frameCopy)
            self.translatorCameraView.setPixmap(self.pixmap)
        if self.capturing and self.currentGesture:
            self.gesture = self.currentGesture
            self.gestureDir = self.modelDir / self.gesture
            self.gestureDir.mkdir(parents=True, exist_ok=True)
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            self.filename = self.gestureDir / f"{self.timestamp}.jpg"
            cv2.imwrite(str(self.filename), self.frameCopy)

    def toggleCapture(self):
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
            self.startCaptureBtn.setText("Start Capture")
            self.updateAllImageCounts()
            self.refreshGestures()
        self.state = "ON" if self.capturing else "OFF"
        self.logStatus(f"Capture {self.state} for '{self.gesture}' gesture", LogLevel.INFO)

    def visualizeModel(self):
        self.logStatus(DATASET_PATH, LogLevel.DEBUG)
        self.labels = []
        for i in os.listdir(DATASET_PATH):
            if os.path.isdir(os.path.join(DATASET_PATH, i)):
                self.labels.append(i)
        self.logStatus(self.labels, LogLevel.INFO)
        for self.label in self.labels:
            self.labelDir = os.path.join(DATASET_PATH, self.label)
            self.exampleFilenames = os.listdir(self.labelDir)[:NUM_EXAMPLES]
            self.fig, self.axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
            for i in range(NUM_EXAMPLES):
                self.axs[i].imshow(plt.imread(os.path.join(self.labelDir, self.exampleFilenames[i])))
                self.axs[i].get_xaxis().set_visible(False)
                self.axs[i].get_yaxis().set_visible(False)
            self.fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {self.label}')
        plt.show()
        return self.labels
    
    def runTraining(self):
        SHARED.mkdir(exist_ok=True)
        subprocess.Popen(["docker", "compose", "run", "--rm", "worker"])
        self.logFile = SHARED / "logs" / "worker.log"
        self.logFile.parent.mkdir(parents=True, exist_ok=True)
        self.logFile.write_text("")
        self._logFilePos = 0
        self.sharedDataset = SHARED / "dataset" / self.modelDir.name
        shutil.copytree(self.modelDir, self.sharedDataset, dirs_exist_ok=True)
        self.job = {"dataset": f"dataset/{self.modelDir.name}", "export": f"exports/{self.exportDir.name}"}
        with open(SHARED / "job.json", "w") as f:
            json.dump(self.job, f)
        subprocess.Popen(["docker", "compose", "build", "--no-cache"], cwd=Path(__file__).parent.parent.parent / "deploy")
        subprocess.Popen(["docker", "compose", "up"], cwd=Path(__file__).parent.parent.parent / "deploy")
        self.logStatus("Started training worker", LogLevel.INFO)

    def trainExportModel(self):
        self.runTraining()
        self.logStatus("Starting Model Training!", LogLevel.INFO)
        self.logStatus("This may take anywhere from 30 Seconds to 20 Minutes depending on hardware capabilities and amount of images being trained.", LogLevel.INFO)
        self.resultCheckTimer = qtc.QTimer()
        self.resultCheckTimer.timeout.connect(self.checkWorkerResult)
        self.resultCheckTimer.start(1000)

    def toggleDebugLogging(self, state):
        if state == qtc.Qt.CheckState.Checked:
            self.runtimeLogger.setLevel(LogLevel.DEBUG)
            self.workerLogger.setLevel(LogLevel.DEBUG)
            self.logStatus("Debug logging enabled", LogLevel.Info)
        else:
            self.runtimeLogger.setLevel(LogLevel.INFO)
            self.workerLogger.setLevel(LogLevel.INFO)
            self.logStatus("Debug logging disabled", LogLevel.INFO)

    def checkWorkerResult(self):
        #
        # check this code to see what it does and if it needs to be changed
        #
        self.resultPath = SHARED / "result.json"
        if not self.resultPath.exists():
            return
        with open(self.resultPath) as f:
            self.result = json.load(f)
        self.logStatus(f"Final accuracy: {self.result['accuracy']}", LogLevel.INFO)
        self.resultCheckTimer.stop()
        self.workerLogTimer.stop()
        subprocess.Popen(["docker", "compose", "down", "-v"], cwd=Path(__file__).parent.parent.parent / "deploy")

        """
        DATASET_PATH = self.modelDir
        data = mp.gesture_recognizer.Dataset.from_folder(dirname=DATASET_PATH, hparams=mp.gesture_recognizer.HandDataPreprocessingParams())
        train_data, rest_data = data.split(0.8)
        validation_data, test_data = rest_data.split(0.5)
        hparams = mp.gesture_recognizer.HParams(exportDir=self.exportDir / "Exported")
        options = mp.gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
        model = mp.gesture_recognizer.GestureRecognizer.create(train_data=train_data, validation_data=validation_data, options=options)
        loss, acc = model.evaluate(test_data, batch_size=1)
        self.logStatus(f"Test loss:{loss}, Test accuracy:{acc}")
        model.export_model()
        print("Exporting Model")
        hparams = mp.gesture_recognizer.HParams(learning_rate=0.003, exportDir=self.exportDir / "Final Export")
        model_options = mp.gesture_recognizer.ModelOptions(dropout_rate=0.2)
        options = mp.gesture_recognizer.GestureRecognizerOptions(model_options=model_options, hparams=hparams)
        model_2 = mp.gesture_recognizer.GestureRecognizer.create(train_data=train_data, validation_data=validation_data, options=options)
        loss, accuracy = model_2.evaluate(test_data)
        self.logStatus(f"Test loss:{loss}, Test accuracy:{accuracy}")
        self.logStatus("Model Exported Successfully")
        print("Model Exported Successfully")
        """
    def openVersionFolder(self):
        os.startfile(self.modelDir)
    def settingsTabUI(self):
        #
        # comepletely rewrite this the base can stay but the actual settings logic needs to be changed
        #
        self.settingsTab = qtw.QWidget()
        self.settingsTabLayout = qtw.QGridLayout()
        self.debugCheckbox = qtw.QCheckBox("Enable debug logging")
        self.settingsTabLayout.addWidget(self.debugCheckbox, 0, 0)
        self.debugCheckbox.stateChanged.connect(self.toggleDebugLogging)
        #self.setTemplate = qtw.QLineEdit()
        #self.setTemplate.setPlaceholderText("Set: ")
        #self.settingsTabLayout.addWidget(self.setTemplate, 0, 0)
        #self.setTemplate.setCheckable(True)
        #self.setTemplate.clicked.connect(self.setTemplater)
        self.settingsTab.setLayout(self.settingsTabLayout)
        return self.settingsTab
    def setTemplater(self):
        print("configure a settings placeholder")
        # example for how to set settings
        # ----------------------------------
        # self.settings.set("modelDir", "D:/NewGesturePath")
        # self.settings.set("darkMode", True)
    def errorMenu(self, message):
        qtw.QMessageBox.critical(self, "Error: ", message, qtw.QMessageBox.StandardButton.Ok)
    def logStatus(self, message, level=LogLevel.INFO):
        self.runtimeLogger.log(message, level)
    def closeProgram(self):
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
        subprocess.Popen(["docker-compose", "down", "-v"], cwd=Path(__file__).parent.parent.parent / "deploy")
        qtw.QApplication.quit()
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec())