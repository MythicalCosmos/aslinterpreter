import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
from core.constants import SETTINGS
from ui.style import ACCENT, TEXT0, TEXT1, TEXT2, BG2


def build_settings_tab(window) -> qtw.QWidget:
    tab = qtw.QWidget()
    scroll = qtw.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(qtw.QFrame.Shape.NoFrame)

    inner = qtw.QWidget()
    layout = qtw.QVBoxLayout(inner)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(24)

    # ── Display ────────────────────────────────────────────────────────────
    layout.addWidget(_group_label("DISPLAY"))
    disp = qtw.QFormLayout()
    disp.setSpacing(10)

    window.windowModeMenu = qtw.QComboBox()
    window.windowModeMenu.addItems(["Windowed", "Fullscreen", "Borderless Fullscreen"])
    window.windowModeMenu.setCurrentText(SETTINGS.app.fullscreen_mode)
    disp.addRow("Window Mode", window.windowModeMenu)

    window.monitorMenu = qtw.QComboBox()
    for i, screen in enumerate(qtw.QApplication.screens()):
        s = screen.size()
        window.monitorMenu.addItem(f"Monitor {i} — {s.width()}×{s.height()}", i)
    window.monitorMenu.setCurrentIndex(SETTINGS.app.monitor)
    disp.addRow("Monitor", window.monitorMenu)

    window.aspectRatioScaleMenu = qtw.QComboBox()
    window.aspectRatioScaleMenu.addItems(["16:9", "4:3"])
    disp.addRow("Aspect Ratio", window.aspectRatioScaleMenu)

    window.resolutionMenu = qtw.QComboBox()
    disp.addRow("Resolution", window.resolutionMenu)

    window.dpiCheck = qtw.QCheckBox("Enable DPI Scaling")
    window.dpiCheck.setChecked(SETTINGS.app.dpi_scaling)
    disp.addRow("", window.dpiCheck)

    layout.addLayout(disp)
    layout.addWidget(_divider())

    # ── Camera ────────────────────────────────────────────────────────────
    layout.addWidget(_group_label("CAMERA"))
    cam = qtw.QFormLayout()
    cam.setSpacing(10)

    window.cameraMenu = qtw.QComboBox()
    cam.addRow("Camera", window.cameraMenu)
    layout.addLayout(cam)
    layout.addWidget(_divider())

    # ── Gesture Recognition ────────────────────────────────────────────────
    layout.addWidget(_group_label("GESTURE RECOGNITION"))
    gest = qtw.QFormLayout()
    gest.setSpacing(10)

    window.confidenceThresholdInput = qtw.QDoubleSpinBox()
    window.confidenceThresholdInput.setRange(0.0, 1.0)
    window.confidenceThresholdInput.setSingleStep(0.05)
    window.confidenceThresholdInput.setValue(SETTINGS.settings.confidence_threshold)
    gest.addRow("Min Word Confidence", window.confidenceThresholdInput)

    window.AutocorrectToggleInput = qtw.QCheckBox("Enable Autocorrect")
    window.AutocorrectToggleInput.setChecked(SETTINGS.settings.autocorrect)
    gest.addRow("", window.AutocorrectToggleInput)

    window.autocorrectThresholdInput = qtw.QDoubleSpinBox()
    window.autocorrectThresholdInput.setRange(0.5, 1.0)
    window.autocorrectThresholdInput.setSingleStep(0.05)
    window.autocorrectThresholdInput.setValue(SETTINGS.settings.autocorrect_threshold)
    gest.addRow("Autocorrect Cutoff", window.autocorrectThresholdInput)

    window.wordGapInput = qtw.QDoubleSpinBox()
    window.wordGapInput.setRange(0.3, 5.0)
    window.wordGapInput.setSingleStep(0.1)
    window.wordGapInput.setValue(SETTINGS.settings.word_gap)
    gest.addRow("Word Gap (seconds)", window.wordGapInput)

    window.linesCheckBoxInput = qtw.QCheckBox("Show Hand Landmark Overlay")
    window.linesCheckBoxInput.setChecked(SETTINGS.settings.lines)
    gest.addRow("", window.linesCheckBoxInput)

    window.previewToggleInput = qtw.QCheckBox("Show Live Letter Preview")
    window.previewToggleInput.setChecked(SETTINGS.settings.preview_toggle)
    gest.addRow("", window.previewToggleInput)

    window.confidenceToggleInput = qtw.QCheckBox("Show Confidence Scores")
    window.confidenceToggleInput.setChecked(SETTINGS.settings.confidence_toggle)
    gest.addRow("", window.confidenceToggleInput)

    layout.addLayout(gest)
    layout.addWidget(_divider())

    # ── Audio Transcription ────────────────────────────────────────────────
    layout.addWidget(_group_label("AUDIO TRANSCRIPTION"))
    audio = qtw.QFormLayout()
    audio.setSpacing(10)

    window.sampleRateInput = qtw.QSpinBox()
    window.sampleRateInput.setRange(8000, 48000)
    window.sampleRateInput.setValue(SETTINGS.settings.sam_rate)
    audio.addRow("Sample Rate (Hz)", window.sampleRateInput)

    window.initialChunkDerationInput = qtw.QDoubleSpinBox()
    window.initialChunkDerationInput.setRange(1.0, 30.0)
    window.initialChunkDerationInput.setValue(SETTINGS.settings.init_chunk_der)
    audio.addRow("Initial Chunk (s)", window.initialChunkDerationInput)

    window.minimumChunkDerationInput = qtw.QDoubleSpinBox()
    window.minimumChunkDerationInput.setRange(0.5, 10.0)
    window.minimumChunkDerationInput.setValue(SETTINGS.settings.min_chunk_der)
    audio.addRow("Minimum Chunk (s)", window.minimumChunkDerationInput)

    window.chunkDecrementInput = qtw.QDoubleSpinBox()
    window.chunkDecrementInput.setRange(0.1, 5.0)
    window.chunkDecrementInput.setValue(SETTINGS.settings.chunk_dec)
    audio.addRow("Chunk Decrement", window.chunkDecrementInput)

    layout.addLayout(audio)
    layout.addWidget(_divider())

    # ── HuggingFace / Diarization ──────────────────────────────────────────
    layout.addWidget(_group_label("DIARIZATION (OPTIONAL)"))
    hf_note = qtw.QLabel(
        "Speaker diarization requires a free HuggingFace account.\n"
        "Get your token at: huggingface.co/settings/tokens"
    )
    hf_note.setStyleSheet(f"color: {TEXT1}; font-size: 12px;")
    hf_note.setWordWrap(True)
    layout.addWidget(hf_note)

    hf_form = qtw.QFormLayout()
    window.hfTokenInput = qtw.QLineEdit()
    window.hfTokenInput.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
    window.hfTokenInput.setEchoMode(qtw.QLineEdit.EchoMode.Password)
    window.hfTokenInput.setText(SETTINGS.env.hf_token)
    hf_form.addRow("HF Token", window.hfTokenInput)
    layout.addLayout(hf_form)
    layout.addWidget(_divider())

    # ── Logging ────────────────────────────────────────────────────────────
    layout.addWidget(_group_label("LOGGING"))
    log_form = qtw.QFormLayout()
    window.logLevelInput = qtw.QComboBox()
    window.logLevelInput.addItems(["Debug", "Info", "Warning", "Error"])
    window.logLevelInput.setCurrentIndex(SETTINGS.app.log_level)
    window.debugCheckbox = qtw.QCheckBox("Enable Debug Logging")
    log_form.addRow("Log Level", window.logLevelInput)
    log_form.addRow("", window.debugCheckbox)
    layout.addLayout(log_form)
    layout.addWidget(_divider())

    # ── Actions ────────────────────────────────────────────────────────────
    btn_row = qtw.QHBoxLayout()

    save_btn = qtw.QPushButton("💾  Save Settings")
    save_btn.setObjectName("primaryBtn")
    save_btn.clicked.connect(window.updateSettings)
    btn_row.addWidget(save_btn)

    reset_btn = qtw.QPushButton("↺  Reset to Defaults")
    reset_btn.clicked.connect(window.confirmResetSettings)
    btn_row.addWidget(reset_btn)

    btn_row.addStretch()
    layout.addLayout(btn_row)

    layout.addStretch()

    scroll.setWidget(inner)
    outer = qtw.QVBoxLayout()
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(scroll)
    container = qtw.QWidget()
    container.setLayout(outer)
    return container


def _group_label(text: str) -> qtw.QLabel:
    lbl = qtw.QLabel(text)
    lbl.setObjectName("sectionHeader")
    return lbl


def _divider() -> qtw.QFrame:
    line = qtw.QFrame()
    line.setFrameShape(qtw.QFrame.Shape.HLine)
    line.setStyleSheet(f"color: #222636; background: #222636; max-height: 1px;")
    return line