#!/usr/bin/env python3
#from mediapipe_model_maker import gesture_recognizer
#assert tf.__version__.startswith('2')
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from datetime import datetime
import PyQt6.QtWidgets as qtw
import ttkbootstrap as tb
import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg
from pathlib import Path
#import tensorflow as tf
#import mediapipe as mp
import threading
import sqlite3
import shutil
import time
import sys
import cv2
import re
import os



DB_FILE = Path(__file__).parent / "gestures.db"
DATASET_PATH = Path(r"\SoftwareDev\exports\data")
EXPORT_PATH = Path(r"\SoftwareDev\exports\model")
DATASET_PATH.mkdir(parents=True, exist_ok=True)
EXPORT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "gestures.task"
CAMERA_INDEX = 0
NUM_EXAMPLES = 5
SETTINGS_FILE = Path(__file__).parent / "settings.txt"


class SettingsManager:
    def __init__(self, filename=SETTINGS_FILE):
        self.filename = filename
        self.settings = {}
        self.load()
    def load(self):
        if not os.path.exists(self.filename):
            return

        with open(self.filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                self.settings[key] = value
    def get(self, key, default=None):
        return self.settings.get(key, default)
    def set(self, key, value):
        self.settings[key] = str(value)
    def save(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            for key, value in self.settings.items():
                f.write(f"{key}={value}\n")

def getNextVersionDir(DATASET_PATH: Path) -> Path:
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    versionRe = re.compile(r"version_(\d+)$")
    highest = 0
    for d in DATASET_PATH.iterdir():
        if d.is_dir() and versionRe.match(d.name):
            num = int(versionRe.match(d.name).group(1))
            highest = max(highest, num)
    nextNum = highest + 1
    versionDir = DATASET_PATH / f"version_{nextNum}"
    return versionDir
def getNextVersionDirExport(EXPORT_PATH: Path) -> Path:
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    versionRe = re.compile(r"version_(\d+)$")
    highest = 0
    for d in EXPORT_PATH.iterdir():
        if d.is_dir() and versionRe.match(d.name):
            num = int(versionRe.match(d.name).group(1))
            highest = max(highest, num)
    nextNum = highest + 1
    exportDir = EXPORT_PATH / f"version_{nextNum}"
    return exportDir

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
        self.dbStartupCheck()
        self.settings = SettingsManager()
        
        self.title = str(self.settings.get("appTitle"))
        self.width = int(self.settings.get("width"))
        self.height = int(self.settings.get("height"))
        self.setWindowTitle(self.title)
        self.resize(self.width, self.height)


        self.versionDir = getNextVersionDir(DATASET_PATH)
        self.exportDir = getNextVersionDirExport(EXPORT_PATH)
        self.datasetPath = DATASET_PATH
        self.exportPath = EXPORT_PATH
        self.exampleAmount = NUM_EXAMPLES
        self.cameraViewLayout = qtw.QVBoxLayout()
        self.statusLayout = qtw.QVBoxLayout()
        self.capturing = False
        self.current_gesture = None
        os.makedirs(self.versionDir, exist_ok=True)
        self.cameraViewLabel = qtw.QLabel()
        self.cameraViewLayout.addWidget(self.cameraViewLabel)
        self.frameTimer = qtc.QTimer()
        self.frameTimer.timeout.connect(self.updateFrame)
        self.frameTimer.start(16)
        self.outLayout = qtw.QVBoxLayout()
        self.centralWid = qtw.QWidget()
        self.centralWid.setLayout(self.outLayout)
        self.setCentralWidget(self.centralWid)
        self.tabs = qtw.QTabWidget()
        self.quitTab = qtw.QWidget()
        self.frame = None
        self.tabs.addTab(self.translatorTabUI(), "Translator")
        self.tabs.addTab(self.modelMakerTabUI(), "Model Maker")
        self.tabs.addTab(self.settingsTabUI(), "Settings")
        self.outLayout.addWidget(self.tabs)
        self.statusOutput = qtw.QTextEdit()
        self.statusOutput.setReadOnly(True)
        self.stdoutRedirector = TextRedirector(self.statusOutput)
        self.stderrRedirector = TextRedirector(self.statusOutput)
        sys.stdout = self.stdoutRedirector
        sys.stderr = self.stderrRedirector
        self.initCamera()
        if self.cap:
            self.launchCameraThread()
            self.updateFrame()
        self.gestures = self.loadExistingGestures(orderByName=True)
        self.loadExistingGestures(orderByName=True)

    def translatorTabUI(self):
        self.modelMakerTab = qtw.QWidget()
        self.modelMakerTabLayout = qtw.QGridLayout()

        self.startCaptureBtn = qtw.QPushButton("Start Capture")
        self.modelMakerTabLayout.addWidget(self.startCaptureBtn, 0, 0)
        self.startCaptureBtn.setCheckable(True)
        #self.startCaptureBtn.clicked.connect(self.startCapture)

        self.refreshGesturesBtn = qtw.QPushButton("Refresh Gestures")
        self.modelMakerTabLayout.addWidget(self.refreshGesturesBtn, 0, 1)

        self.visualizeModelBtn = qtw.QPushButton("Visualize Model")
        self.modelMakerTabLayout.addWidget(self.visualizeModelBtn, 0, 2)

        self.trainExportModelBtn = qtw.QPushButton("Train & Export Model")
        self.modelMakerTabLayout.addWidget(self.trainExportModelBtn, 0, 3)

        self.versionFolderBtn = qtw.QPushButton("Open Version Folder")
        self.modelMakerTabLayout.addWidget(self.versionFolderBtn, 0, 4)

        self.deleteGestureBtn = qtw.QPushButton("Delete Selected Gesture")
        self.modelMakerTabLayout.addWidget(self.deleteGestureBtn, 0, 5)
        
        self.modelMakerTab.setLayout(self.modelMakerTabLayout)
        return self.modelMakerTab
    
    def modelMakerTabUI(self):
        #
        # Layouts: 
        # modelMakerTabLayout() is the main layout for the tab so it's just the outer ring for the tab 
        # outerGestureControlTreeBtnLayout() is the layout for the gesture management section of this tab organizes the whole gesture management section
        # gestureControlTreeBtnLayout() is the layout for the buttons that control the gesture management just used for convienece for organization
        # gestureControlTreeModelViewAndViewLayout() is the layout for showing what gesture exist and the information tied to them
        #
        #
        self.modelMakerTab = qtw.QWidget()
        self.modelMakerTabLayout = qtw.QVBoxLayout()
        self.treeAndCameraLayout = qtw.QHBoxLayout()
        self.outerGestureControlTreeBtnLayout = qtw.QVBoxLayout()
        self.gestureControlTreeBtnLayout = qtw.QHBoxLayout()
        self.gestureControlTreeModelViewAndViewLayout = qtw.QHBoxLayout()
        self.gestureNameInput = qtw.QLineEdit()
        self.gestureNameInput.setPlaceholderText("New Gesture Name: ")
        self.gestureControlTreeBtnLayout.addWidget(self.gestureNameInput, 0)
        self.cameraViewLayout.addWidget(self.cameraViewLabel, 0)
        self.addGestureBtn = qtw.QPushButton("Add")
        self.gestureControlTreeBtnLayout.addWidget(self.addGestureBtn, 1)
        self.addGestureBtn.clicked.connect(self.gestureNameExistsCheck)
        self.deleteGestureBtn = qtw.QPushButton("Delete Gesture",)
        self.gestureControlTreeBtnLayout.addWidget(self.deleteGestureBtn, 2)
        self.deleteGestureBtn.clicked.connect(self.gestureSelectedCheck)
        self.statusOutput.setFixedHeight(120)
        self.statusOutput.setLineWrapMode(qtw.QTextEdit.LineWrapMode.WidgetWidth)
        self.statusLayout.addWidget(self.statusOutput)
        self.statusFrame = qtw.QWidget()

        self.startCaptureBtn = qtw.QPushButton("Start Capture")
        self.modelMakerTabLayout.addWidget(self.startCaptureBtn, 0)
        self.startCaptureBtn.setCheckable(True)
        self.startCaptureBtn.clicked.connect(self.startCapture)

        self.refreshGesturesBtn = qtw.QPushButton("Refresh Gestures")
        self.modelMakerTabLayout.addWidget(self.refreshGesturesBtn, 0, 1)
        self.refreshGesturesBtn.clicked.connect(self.refreshGestures)

        self.visualizeModelBtn = qtw.QPushButton("Visualize Model")
        self.modelMakerTabLayout.addWidget(self.visualizeModelBtn, 0, 2)
        self.visualizeModelBtn.clicked.connect(self.visualizeModel)

        self.trainExportModelBtn = qtw.QPushButton("Train & Export Model")
        self.modelMakerTabLayout.addWidget(self.trainExportModelBtn, 0, 3)
        self.trainExportModelBtn.clicked.connect(self.trainExportModel)

        self.versionFolderBtn = qtw.QPushButton("Open Version Folder")
        self.modelMakerTabLayout.addWidget(self.versionFolderBtn, 0, 4)
        self.versionFolderBtn.clicked.connect(self.openVersionFolder)

        self.quitProgramBtn = qtw.QPushButton("Quit Program")
        self.modelMakerTabLayout.addWidget(self.quitProgramBtn, 0, 5)
        self.quitProgramBtn.clicked.connect(self.closeProgram)
        
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
        self.listGesturesTree.itemClicked.connect(self.whichGestureSelected)
        self.gestureData = []
        


        self.gestureControlTreeLabel = qtw.QLabel("Gesture Management")
        self.statusFrame.setLayout(self.statusLayout)
        self.outLayout.addWidget(self.statusFrame)
        self.outerGestureControlTreeBtnLayout.addWidget(self.gestureControlTreeLabel, 0)
        self.outerGestureControlTreeBtnLayout.addLayout(self.gestureControlTreeBtnLayout, 1)
        self.outerGestureControlTreeBtnLayout.addLayout(self.gestureControlTreeModelViewAndViewLayout, 2)
        self.treeAndCameraLayout.addLayout(self.outerGestureControlTreeBtnLayout, 0)
        self.modelMakerTabLayout.addLayout(self.treeAndCameraLayout, 1)
        self.modelMakerTab.setLayout(self.modelMakerTabLayout)
        return self.modelMakerTab
    
    def gestureNameExistsCheck(self):
        if self.gestureNameInput.text().strip():
            self.addGesture()
        else:
            self.errorMenu(message="The Gesture Does Not Have a Name.")

    def loadExistingGestures(self, orderByName=True):
        self.listGesturesTree.clear()
        self.connect = sqlite3.connect(DB_FILE)
        self.cursor = self.connect.cursor()
        self.query = "SELECT name, count FROM gestures"
        if orderByName:
            self.query += " ORDER BY name COLLATE NOCASE"
        self.cursor.execute(self.query)
        self.gestures = self.cursor.fetchall()
        self.connect.close()
        return [{"name": g[0], "count": g[1]} for g in self.gestures]
    
    def closeEvent(self, event):
        self.closeProgram()
        event.accept()
    
    def addGesture(self):
        self.name = self.gestureNameInput.text().strip()
        if not self.name:
            return
        self.addGestureToDatabase(self.name)
        self.gestureNameInput.clear()
        self.refreshGestures()
    
    def updateGestureCount(self, name, count):
        self.connect = sqlite3.connect(DB_FILE)
        self.cursor = self.connect.cursor()
        self.cursor.execute("UPDATE gestures SET count = ? WHERE name = ?", (count, name))
        self.connect.commit()
        self.connect.close()

    def addGestureToDatabase(self, name, count=CAMERA_INDEX):
        self.connect = sqlite3.connect(DB_FILE)
        self.cursor = self.connect.cursor()
        self.cursor.execute("""INSERT OR IGNORE INTO gestures (name, count) VALUES (?, ?)""", (name, count))
        self.connect.commit()
        self.connect.close()

    def gestureInfo(self, item):
        self.name = self.item.text(0)
        self.gestures = self.loadExistingGestures(order_by_name=False)
        self.gesture = next((g for g in self.gestures if g["name"] == self.name), None)
        if self.gesture:
            self.listGesturesTree.clear()
            self.row = qtw.QTreeWidgetItem([self.gesture["name"], str(self.gesture["count"])])
            self.info_tree.addTopLevelItem(self.row)

    def refreshGestures(self):
        self.listGesturesTree.clear()
        self.gestures = self.loadExistingGestures(orderByName=True)
        for g in self.gestures:
            item = qtw.QTreeWidgetItem([g["name"]])
            self.listGesturesTree.addTopLevelItem(item)

    def gestureSelectedCheck(self):
        self.item = self.selectedGesture()
        if self.item:
            self.confirmGestureDelete(self.item)
        else:
            self.errorMenu(message="A Gesture is Not Selected.")
            
    def selectedGesture(self):
         return self.listGesturesTree.currentItem()

    def confirmGestureDelete(self, item):
        name = self.item.text(0)
        self.reply = qtw.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the gesture: '{name}'?",
            qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No,
            qtw.QMessageBox.StandardButton.No)
        if self.reply == qtw.QMessageBox.StandardButton.Yes:
            self.deleteGesture(item, name)
        else:
            return

    def deleteGesture(self, item, name):
        self.connect = sqlite3.connect(DB_FILE)
        self.cursor = self.connect.cursor()
        self.cursor.execute("DELETE FROM gestures WHERE name = ?", (name,))
        self.connect.commit()
        self.connect.close()
        self.parent = self.item.parent()
        if self.parent is None:
            self.index = self.listGesturesTree.indexOfTopLevelItem(self.item)
            self.listGesturesTree.takeTopLevelItem(self.index)
        else:
            self.parent.removeChild(self.item)

        del self.item
        print("Deleting:", name)

    def whichGestureSelected(self):
        self.selectedGesture()
    
    def startCapture(self):
        print("Starts capture")

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
        item = self.listGesturesTree.currentItem()
        if not item:
            showFrameOnly = True
        self.name = item.text(0)
        frameCopy = None
        with self.frameLock:
            if self.frame is not None:                    
                frameCopy = self.frame.copy()
        if frameCopy is not None:
            rgb = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytesPerLine = ch * w
            qimg = qtg.QImage(rgb.data, w, h, bytesPerLine, qtg.QImage.Format.Format_RGB888)
            pixmap = qtg.QPixmap.fromImage(qimg)
            self.cameraViewLabel.setPixmap(pixmap)
            if self.capturing and self.name:
                gestureDir = os.path.join(self.versionDir, self.name)
                os.makedirs(gestureDir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                filename = os.path.join(gestureDir, f"{timestamp}.jpg")
                cv2.imwrite(filename, frameCopy)
                imageCount = len([
                    f for f in os.listdir(gestureDir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                self.updateGestureCount(self.name, imageCount)
                if imageCount % 500 == 0:
                    self.logStatus(f"Captured {imageCount} images for '{self.name}'")
    def toggleCapture(self):
        if not self.cap:
            self.errorMenu(message="No camera available.")
            return
        self.item = self.listGesturesTree.currentItem()
        if not self.item:
            self.errorMenu("No gesture selected")
            return
        self.sel = self.item.text(0)
        gesture = self.sel
        self.currentGesture = self.sel
        self.capturing = not self.capturing
        if self.capturing:
            self.startCaptureBtn.setText("Stop Capture")
            gestureDir = os.path.join(self.versionDir, gesture)
            os.makedirs(gestureDir, exist_ok=True)
            existingCount = len(os.listdir(gestureDir))
            self.logStatus(f"Started Capture for gesture '{gesture}'.")
        else:
            self.startCaptureBtn.setText("Start Capture")
            currentCount = self.gestures.get(gesture, 0)
            self.logStatus(f"Stopped capture for '{gesture}' (total: {currentCount} images)")
            self.refreshGestures()
        state = "ON" if self.capturing else "OFF"
        self.logStatus(f"Capture {state} for '{gesture}' gesture")


    def visualizeModel(self):
        DATASET_PATH = self.versionDir
        print(DATASET_PATH)
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

    def trainExportModel(self):
        print("Trains and Exports the model")

    def openVersionFolder(self):
        os.startfile(self.versionDir)



        
    def settingsTabUI(self):
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
        # self.settings.set("versionDir", "D:/NewGesturePath")
        # self.settings.set("darkMode", True)

    def saveAll(self):
        #add a confirm save all settings menu
        self.settings.save()

    def dbStartupCheck(self):
        self.connect = sqlite3.connect(DB_FILE)
        self.cursor = self.connect.cursor()
        self.cursor.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name='gestures';""")
        self.tableExists = self.cursor.fetchone() is not None
        if self.tableExists:
            print("Table 'gestures' exists")
        else:
            self.cursor.execute("""CREATE TABLE IF NOT EXISTS gestures (name TEXT PRIMARY KEY, count INTEGER)""")
        self.connect.commit()
        self.connect.close()

    def errorMenu(self, message):
        qtw.QMessageBox.critical(
        self,
        "Error: ",
        message,
        qtw.QMessageBox.StandardButton.Ok
    )
    
    def log_status(self, message):
        print(message) 
    
    def closeProgram(self):
        if hasattr(self, "stopEvent"):
            self.stopEvent.set()
        if hasattr(self, "cameraThread") and self.cameraThread:
            self.cameraThread.join(timeout=1)
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        self.settings.set("width", self.width())
        self.settings.set("height", self.height())
        self.settings.save()

        qtw.QApplication.quit()

        


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec())