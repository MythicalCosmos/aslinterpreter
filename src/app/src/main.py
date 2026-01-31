from config.loader import load_settings
#!/usr/bin/env python3
#
#
# Fuck you script it sucks dick to debug 
#
#
#from mediapipe_model_maker import gesture_recognizer as mp
#assert tf.__version__.startswith('2')
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
SHARED = JSON_FILE =  Path(__file__).parent.parent.parent / "shared"
DB_FILE = Path(__file__).parent / "gestures.db"
DATASET_PATH = Path(__file__).parent.parent.parent / "shared/dataset"
EXPORT_PATH = Path(__file__).parent.parent.parent / "shared/exports"
#EXPORT_PATH.mkdir(parents=True, exist_ok=True)
#DATASET_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = SETTINGS.gestures.gesture_model
CAMERA_INDEX = 0
NUM_EXAMPLES = SETTINGS.settings.examples
CONFIG_PATH = None
JSON_FILE = JSON_FILE =  Path(__file__).parent.parent.parent / "shared/gestures.json"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
BASE_DATA = {
    "name": "", "image_count": ""
}
class TextRedirector(qtc.QObject):
    textWritten = qtc.pyqtSignal(str)
    def __init__(self, textEdit: qtw.QTextEdit, mirrorToTerminal=True):
        super().__init__()
        self.textEdit = textEdit
        self.mirrorToTerminal = mirrorToTerminal
        self._stdout = sys.__stdout__
        self._stderr = sys.__stderr__
        self.textWritten.connect(self._append_text)
    def write(self, message):
        if not message.strip():
            return
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        fullMessage = timestamp + message.rstrip()
        self.textWritten.emit(fullMessage)
        if self.mirrorToTerminal:
            self._stdout.write(fullMessage + "\n")
            self._stdout.flush()
    def flush(self):
        pass
    @qtc.pyqtSlot(str)
    def _append_text(self, message):
        self.textEdit.append(message)
class MainGui(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = SETTINGS.app.name
        self.width = SETTINGS.app.width
        self.height = SETTINGS.app.height
        self.setWindowTitle(self.title)
        print("Window size:", self.width, self.height)
        self.resize(self.width, self.height)
        if self.width <= 0 or self.height <= 0:
            self.resize(1280, 800)
        self.modelDir = Path(DATASET_PATH)
        self.exportDir = Path(EXPORT_PATH)
        self.datasetPath = Path(DATASET_PATH)
        self.exportPath = Path(EXPORT_PATH)
        self.exampleAmount = NUM_EXAMPLES
        self.cameraViewLayout = qtw.QVBoxLayout()
        self.translatorCameraViewLayout = qtw.QVBoxLayout()
        self.statusLayout = qtw.QVBoxLayout()
        self.capturing = False
        self.currentGesture = None
        os.makedirs(self.modelDir, exist_ok=True)
        self.workerLogPath = Path(__file__).parent.parent.parent / "shared/logs/worker.log"
        self._logFilePos = 0
        self.workerLogTimer = qtc.QTimer()
        #self.workerLogTimer.timeout.connect(self.readWorkerLogs)
        self.workerLogTimer.start(250)
        self.translatorCameraLabel = qtw.QLabel()
        self.translatorCameraViewLayout.addWidget(self.translatorCameraLabel)
        self.cameraViewLabel = qtw.QLabel()
        self.cameraViewLayout.addWidget(self.cameraViewLabel)
        self.outLayout = qtw.QVBoxLayout()
        self.centralWid = qtw.QWidget()
        self.centralWid.setLayout(self.outLayout)
        self.setCentralWidget(self.centralWid)
        self.tabs = qtw.QTabWidget()
        self.quitTab = qtw.QWidget()
        self.statusOutput = qtw.QTextEdit()
        self.statusOutput.setReadOnly(True)
        self.stdoutRedirector = TextRedirector(self.statusOutput)
        self.stderrRedirector = TextRedirector(self.statusOutput)
        sys.stdout = self.stdoutRedirector
        sys.stderr = self.stderrRedirector
        self.frame = None
        self.tabs.addTab(self.translatorTabUI(), "Translator")
        self.tabs.addTab(self.modelMakerTabUI(), "Model Maker")
        self.tabs.addTab(self.settingsTabUI(), "Settings")
        self.outLayout.addWidget(self.tabs, 0)
        self.initCamera()
        self.gestures = self.loadExistingGestures(orderByName=True)
        self.loadExistingGestures(orderByName=True)
        self.frameTimer = qtc.QTimer()
        self.frameTimer.timeout.connect(self.updateFrame)
        self.frameTimer.start(16)
        if self.cap:
            self.launchCameraThread()
            self.updateFrame()
    def translatorTabUI(self):
        self.translatorTab = qtw.QWidget()
        self.translatorTabLayout = qtw.QGridLayout()
        self.translatorTabLayout.addLayout(self.translatorCameraViewLayout, 0, 0)

        
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
        self.statusLayout.addWidget(self.statusOutput)
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
        #self.listGesturesTree.itemClicked.connect(self.whichGestureSelected)
        self.gestureData = []
        self.gestureControlTreeLabel = qtw.QLabel("Gesture Management")
        self.statusFrame.setLayout(self.statusLayout)
        self.modelMakerTabLayout.addWidget(self.statusFrame, 3, 0, 1, -1)
        self.outerGestureControlTreeBtnLayout.addWidget(self.gestureControlTreeLabel, 0)
        self.outerGestureControlTreeBtnLayout.addLayout(self.gestureControlTreeBtnLayout, 1)
        self.treeAndCameraLayout.addLayout(self.outerGestureControlTreeBtnLayout)
        self.treeAndCameraLayout.addLayout(self.cameraViewLayout)
        self.outerGestureControlTreeBtnLayout.addLayout(self.gestureControlTreeModelViewAndViewLayout, 2)
        self.modelMakerTabLayout.addLayout(self.treeAndCameraLayout, 1, 0)
        self.modelMakerTabLayout.addWidget(self.datasetPathLabel, 4, 0, 1, -1)
        self.modelMakerTab.setLayout(self.modelMakerTabLayout)
        self.cameraViewLabel.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.listGesturesTree.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.gestureTreeInfo.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.statusOutput.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.cameraViewLabel.setScaledContents(True)
        #self.cameraViewLabel.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.modelMakerTabLayout.setRowStretch(0, 0)
        #self.modelMakerTabLayout.setRowStretch(1, 1)
        self.modelMakerTabLayout.setRowStretch(3, 0)
        self.modelMakerTabLayout.setRowStretch(4, 0)
        self.modelMakerTabLayout.setColumnStretch(0, 1)
        self.treeAndCameraLayout.setStretch(0, 1)
        self.treeAndCameraLayout.setStretch(1, 2)
        self.gestureTreeInfo.header().setSectionResizeMode(qtw.QHeaderView.ResizeMode.Stretch)
        self.listGesturesTree.header().setStretchLastSection(True)
        self.statusFrame.setSizePolicy(qtw.QSizePolicy.Policy.Expanding, qtw.QSizePolicy.Policy.Expanding)
        self.gestureTreeInfo.header().setSectionResizeMode(qtw.QHeaderView.ResizeMode.Stretch)
        self.listGesturesTree.header().setStretchLastSection(True)
        return self.modelMakerTab
    def gestureNameExistsCheck(self):
        if self.gestureNameInput.text().strip():
           self.addGesture(self.gestureNameInput.text().strip())
        else:
            self.errorMenu(message="The Gesture Does Not Have a Name.")
    def loadData(self):
        if not os.path.exists(JSON_FILE):
            return []
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            print("WARNING: gestures.json is not a list — fixing automatically")
            data = [data]
        if not isinstance(data, list):
            print("ERROR: gestures.json is invalid")
            return []

        return data
    def saveData(self, data):
        with open(JSON_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def countImages(self, directory):
        if not os.path.exists(directory):
            return 0
        return sum(1 for file in os.listdir(directory) if file.lower().endswith(IMAGE_EXTENSIONS))
    
    def addGesture(self, name):
        data = self.loadData()
        if any(entry["name"] == name for entry in data):
            print(f"Entry '{name}' already exists.")
            return
        imageDir = Path(DATASET_PATH) / name
        imageCount = self.countImages(imageDir)
        data.append({
            "name": name,
            "image_count": imageCount
        })
        self.saveData(data)
        self.loadExistingGestures()
        self.gestureNameInput.clear()
        self.logStatus(f"Added '{name}' with {imageCount} images.")

    def updateAllImageCounts(self):
        data = self.loadData()
        for entry in data:
            folder = os.path.join(DATASET_PATH, entry["name"])
            entry["image_count"] = self.countImages(folder)
        self.saveData(data)
        print("Image counts updated.")
    
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
        self.logStatus(f"Deleted gesture '{name}'")
        self.name = None
        self.currentGesture = None
        self.capturing = False
        self.frameTimer.start(16)
        self.loadExistingGestures()
    def loadExistingGestures(self, orderByName=True):
        self.listGesturesTree.clear()
        self.gestureTreeInfo.clear()
        data = self.loadData()
        if orderByName:
            data = sorted(data, key=lambda x: x["name"].lower())
        self.gestures = data
        for entry in data:
            name = entry["name"]
            count = entry["image_count"]
            self.listGesturesTree.addTopLevelItem(
            qtw.QTreeWidgetItem([name])
            )
            self.gestureTreeInfo.addTopLevelItem(
                qtw.QTreeWidgetItem([name, str(count)])
            )
        return data
        
    def refreshGestures(self):
        self.loadExistingGestures(orderByName=True)
        self.logStatus("Refreshed Gesture Tree")
    def gestureSelectedCheck(self):
        self.item = self.selectedGesture()
        if self.item:
            self.confirmGestureDelete(self.item)
        else:
            self.errorMenu(message="A Gesture is Not Selected.")  
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
    def confirmGestureDelete(self, item):
        item = self.listGesturesTree.currentItem()
        if not item:
            return
        self.name = item.text(0)
        reply = qtw.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the gesture: '{self.name}'?",
            qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No,
            qtw.QMessageBox.StandardButton.No
        )
        if reply == qtw.QMessageBox.StandardButton.Yes:
            self.frameTimer.stop()
            if self.capturing:
                self.capturing = False
                self.startCaptureBtn.setText("Start Capture")
            self.deleteGesture(self.name)
            self.frameTimer.start(16)
    def startCapture(self):
        self.toggleCapture()
    def initCamera(self):
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
            self.errorMenu(message="No camera found")
            self.cap = None
            return             
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        print(f"Found Camera {self.cap.isOpened()}") 
        self.stopEvent = threading.Event()
        self.frameLock = threading.Lock()
        self.frame = None
        self.cameraThread = None
    def cameraLoop(self):
        while not getattr(self, "stopEvent", threading.Event()).is_set() and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            with self.frameLock:
                self.frame = frame    
            time.sleep(0.01)
    def launchCameraThread(self):
        if getattr(self, "cameraThread", None) and self.cameraThread.is_alive():
            return
        self.stopEvent.clear()
        self.cameraThread = threading.Thread(target=self.cameraLoop, daemon=True)
        self.cameraThread.start()
        print("Camera started") # put it into the bottom terminal
    def updateFrame(self):
        # Determine the current gesture name safely
        if isinstance(self.currentGesture, qtw.QTreeWidgetItem):
            gesture_name = self.currentGesture.text(0)
        elif isinstance(self.currentGesture, str):
            gesture_name = self.currentGesture
        else:
            gesture_name = None
        self.name = gesture_name
        if not self.name:
            # No gesture selected, just update camera preview
            frameCopy = None
            with self.frameLock:
                if self.frame is not None:
                    frameCopy = self.frame.copy()
                    rgb = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    bytesPerLine = ch * w
                    qimg = qtg.QImage(rgb.data, w, h, bytesPerLine, qtg.QImage.Format.Format_RGB888)
                    pixmap = qtg.QPixmap.fromImage(qimg)
                    scaled = pixmap.scaled(
                        self.cameraViewLabel.size(),
                        qtc.Qt.AspectRatioMode.KeepAspectRatio,
                        qtc.Qt.TransformationMode.SmoothTransformation
                    )
                    self.cameraViewLabel.setPixmap(scaled)
            return
        # Ensure directory exists for capturing
        saveDir = self.modelDir / self.name
        if not saveDir.exists():
            saveDir.mkdir(parents=True, exist_ok=True)
        # Copy and display frame
        frameCopy = None
        with self.frameLock:
            if self.frame is not None:
                frameCopy = self.frame.copy()
                rgb = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytesPerLine = ch * w
                qimg = qtg.QImage(rgb.data, w, h, bytesPerLine, qtg.QImage.Format.Format_RGB888)
                # Update both camera labels
                pixmap = qtg.QPixmap.fromImage(qimg)
                scaled = pixmap.scaled(
                    self.cameraViewLabel.size(),
                    qtc.Qt.AspectRatioMode.KeepAspectRatio,
                    qtc.Qt.TransformationMode.SmoothTransformation
                )
                self.cameraViewLabel.setPixmap(scaled)
                scaled2 = pixmap.scaled(
                    self.translatorCameraLabel.size(),
                    qtc.Qt.AspectRatioMode.KeepAspectRatio,
                    qtc.Qt.TransformationMode.SmoothTransformation
                )
                self.translatorCameraLabel.setPixmap(scaled2)
                # Capture image if active
                if self.capturing and self.name:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    gestureDir = self.modelDir / self.name
                    gestureDir.mkdir(parents=True, exist_ok=True)
                    filename = gestureDir / f"{timestamp}.jpg"
                    cv2.imwrite(str(filename), frameCopy)
                    imageCount = len([f for f in os.listdir(gestureDir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    if imageCount % 500 == 0:
                        self.logStatus(f"Captured {imageCount} images for '{self.name}'")
    def toggleCapture(self):
        if not self.cap:
            self.errorMenu(message="No camera available.")
            return
        self.item = self.listGesturesTree.currentItem()
        if not self.item:
            self.errorMenu(message="No gesture selected")
            return
        self.sel = self.item.text(0)
        gesture = self.sel
        self.currentGesture = self.sel
        self.capturing = not self.capturing
        if self.capturing:
            self.startCaptureBtn.setText("Stop Capture")
            gestureDir = os.path.join(self.modelDir, gesture)
            os.makedirs(gestureDir, exist_ok=True)
            self.logStatus(f"Started Capture for gesture '{gesture}'.")
        else:
            self.startCaptureBtn.setText("Start Capture")
            self.updateAllImageCounts()
            self.refreshGestures()
        state = "ON" if self.capturing else "OFF"
        self.logStatus(f"Capture {state} for '{gesture}' gesture")
    def visualizeModel(self):
        self.logStatus(DATASET_PATH)
        self.labels = []
        for i in os.listdir(DATASET_PATH):
            if os.path.isdir(os.path.join(DATASET_PATH, i)):
                self.labels.append(i)
        print(self.labels)
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
        #
        # check this code to see what it actually does
        #
        SHARED.mkdir(exist_ok=True)
        subprocess.Popen(["docker", "compose", "run", "--rm", "worker"])
        # Clear logs
        log_file = SHARED / "logs" / "worker.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("")
        self._logFilePos = 0
        # Copy dataset
        shared_dataset = SHARED / "dataset" / self.modelDir.name
        shutil.copytree(self.modelDir, shared_dataset, dirs_exist_ok=True)
        job = {"dataset": f"dataset/{self.modelDir.name}", "export": f"exports/{self.exportDir.name}"}
        with open(SHARED / "job.json", "w") as f:
            json.dump(job, f)
        subprocess.Popen(["docker", "compose", "build", "--no-cache"], cwd=Path(__file__).parent.parent.parent / "deploy")
        subprocess.Popen(["docker", "compose", "up"], cwd=Path(__file__).parent.parent.parent / "deploy")
        self.logStatus("Started training worker")
    def trainExportModel(self):
        #
        # check what this does and if it needs to be changed
        #
        self.runTraining()
        self.logStatus("Starting Model Training!")
        self.logStatus("This may take anywhere from 30 Seconds to 20 Minutes depending on hardware capabilities and amount of images being trained.")
        self.resultCheckTimer = qtc.QTimer()
        self.resultCheckTimer.timeout.connect(self.checkWorkerResult)
        self.resultCheckTimer.start(1000)
    def checkWorkerResult(self):
        #
        # check this code to see what it does and if it needs to be changed
        #
        self.resultPath = SHARED / "result.json"
        if not self.resultPath.exists():
            return
        with open(self.resultPath) as f:
            result = json.load(f)
        self.logStatus(f"Final accuracy: {result['accuracy']}")
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
        self.setTemplate = qtw.QLineEdit()
        self.setTemplate.setPlaceholderText("Set: ")
        self.settingsTabLayout.addWidget(self.setTemplate, 0, 0)
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
    def logStatus(self, message):
        print(message) 
    def closeProgram(self):
        if hasattr(self, "stopEvent"):
            self.stopEvent.set()
        if hasattr(self, "cameraThread") and self.cameraThread:
            self.cameraThread.join(timeout=1)
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        subprocess.Popen(["docker-compose", "down", "-v"], cwd=Path(__file__).parent.parent.parent / "deploy")
        qtw.QApplication.quit()
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec())