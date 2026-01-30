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
import shutil
import time
import sys
import cv2
import re
import os

class MainGui(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Asl English Converter")
        self.pageLayout = qtw.QGridLayout()
        self.buttonLayout = qtw.QHBoxLayout()
        self.menuLayout = qtw.QStackedLayout()
        self.pageLayout.addLayout(self.buttonLayout, 0, 0)
        self.pageLayout.addLayout(self.menuLayout, 0, 1)
        
        btn = qtw.QPushButton("red")
        btn.pressed.connect(self.activate_tab_1)
        self.buttonLayout.addWidget(btn)
        self.menuLayout.addWidget(qtw.QLabel("red"))

        btn = qtw.QPushButton("green")
        btn.pressed.connect(self.activate_tab_2)
        self.buttonLayout.addWidget(btn)
        self.menuLayout.addWidget(qtw.QLabel("green"))

        btn = qtw.QPushButton("yellow")
        btn.pressed.connect(self.activate_tab_3)
        self.buttonLayout.addWidget(btn)
        self.menuLayout.addWidget(qtw.QLabel("yellow"))

        widget = qtw.QWidget()
        widget.setLayout(self.pageLayout)
        self.setCentralWidget(widget)

    def activate_tab_1(self):
        self.menuLayout.setCurrentIndex(0)

    def activate_tab_2(self):
        self.menuLayout.setCurrentIndex(1)

    def activate_tab_3(self):
        self.menuLayout.setCurrentIndex(2)
    





app = qtw.QApplication(sys.argv)
window = MainGui()
window.show()
app.exec()