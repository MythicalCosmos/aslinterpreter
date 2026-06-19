import os
from datetime import datetime
from pathlib import Path
import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg
from ui.widgets.aspect_ratio_widget import AspectRatioWidget
from ui.widgets.log_viewer import LogViewer
from core.constants import WORKER_LOG_PATH, CLASSIFIER_MODEL_PATH, EXPORT_PATH
from ui.style import ACCENT, ACCENT2, BG1, BG2, BG3, TEXT0, TEXT1, TEXT2


def build_translator_tab(window) -> qtw.QWidget:
    """
    Build and return the translator tab widget.
    `window` is the MainGui instance — used to connect signals.
    """
    tab = qtw.QWidget()
    root = qtw.QHBoxLayout(tab)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(12)

    # ── Left column: camera + controls ────────────────────────────────────
    left = qtw.QVBoxLayout()
    left.setSpacing(8)

    cam_header = _section_label("SIGNING VIEW")
    left.addWidget(cam_header)

    window.translatorCameraView = AspectRatioWidget(16/9)
    window.translatorCameraView.setMinimumSize(400, 225)
    left.addWidget(window.translatorCameraView, stretch=1)

    # Current letter indicator
    letter_row = qtw.QHBoxLayout()
    letter_label = _section_label("CURRENT LETTER")
    window.currentLetterDisplay = qtw.QLabel("—")
    window.currentLetterDisplay.setObjectName("primaryOutput")
    window.currentLetterDisplay.setFixedHeight(60)
    window.currentLetterDisplay.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
    letter_row.addWidget(letter_label)
    letter_row.addWidget(window.currentLetterDisplay, stretch=1)
    left.addLayout(letter_row)

    # Confidence bar
    conf_row = qtw.QHBoxLayout()
    conf_lbl = _section_label("CONFIDENCE")
    window.confidenceBar = qtw.QProgressBar()
    window.confidenceBar.setRange(0, 100)
    window.confidenceBar.setValue(0)
    window.confidenceBar.setFixedHeight(8)
    window.confidenceBar.setTextVisible(False)
    conf_row.addWidget(conf_lbl)
    conf_row.addWidget(window.confidenceBar, stretch=1)
    left.addLayout(conf_row)

    # Audio recording button
    window.audioRecordBtn = qtw.QPushButton("⏺  Start Audio Transcription")
    window.audioRecordBtn.setCheckable(True)
    window.audioRecordBtn.setObjectName("primaryBtn")
    window.audioRecordBtn.clicked.connect(window.toggleAudioRecording)
    left.addWidget(window.audioRecordBtn)

    # Export button
    export_btn = qtw.QPushButton("💾  Export Transcript")
    export_btn.clicked.connect(window.exportTranscript)
    left.addWidget(export_btn)

    # Clear button
    clear_btn = qtw.QPushButton("🗑  Clear All")
    clear_btn.clicked.connect(window.clearTranscript)
    left.addWidget(clear_btn)

    # Path info
    path_lbl = qtw.QLabel(f"Model: {CLASSIFIER_MODEL_PATH.name}")
    path_lbl.setStyleSheet(f"color: {TEXT2}; font-size: 11px;")
    left.addWidget(path_lbl)

    left_widget = qtw.QWidget()
    left_widget.setLayout(left)
    left_widget.setFixedWidth(440)

    # ── Right column: outputs ──────────────────────────────────────────────
    right = qtw.QVBoxLayout()
    right.setSpacing(8)

    # ASL (signed letter stream) output
    right.addWidget(_section_label("SIGNED OUTPUT"))
    window.aslTranscriptionOutput = qtw.QTextEdit()
    window.aslTranscriptionOutput.setReadOnly(True)
    window.aslTranscriptionOutput.setPlaceholderText(
        "Signed letters will appear here as you sign...\n\n"
        "Letters are grouped into words automatically."
    )
    window.aslTranscriptionOutput.setFont(qtg.QFont("JetBrains Mono", 14))
    right.addWidget(window.aslTranscriptionOutput, stretch=3)

    # Audio transcription output
    right.addWidget(_section_label("AUDIO TRANSCRIPTION"))
    window.transcriptionOutput = qtw.QTextEdit()
    window.transcriptionOutput.setReadOnly(True)
    window.transcriptionOutput.setPlaceholderText(
        "Audio transcription will appear here once recording starts..."
    )
    right.addWidget(window.transcriptionOutput, stretch=2)

    # Debug log (collapsible)
    debug_toggle = qtw.QPushButton("▶  Show Debug Log")
    debug_toggle.setCheckable(True)
    debug_toggle.setFixedHeight(30)
    right.addWidget(debug_toggle)

    window.translatorStatusOutput = LogViewer(maxLines=500)
    window.translatorStatusOutput.setMaximumHeight(120)
    window.translatorStatusOutput.setVisible(False)
    right.addWidget(window.translatorStatusOutput)

    def toggle_debug(checked):
        window.translatorStatusOutput.setVisible(checked)
        debug_toggle.setText("▼  Hide Debug Log" if checked else "▶  Show Debug Log")

    debug_toggle.toggled.connect(toggle_debug)

    # ── Assemble ──────────────────────────────────────────────────────────
    root.addWidget(left_widget)
    right_widget = qtw.QWidget()
    right_widget.setLayout(right)
    root.addWidget(right_widget, stretch=1)

    return tab


def _section_label(text: str) -> qtw.QLabel:
    lbl = qtw.QLabel(text)
    lbl.setObjectName("sectionHeader")
    return lbl