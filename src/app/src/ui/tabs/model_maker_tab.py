import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg
from ui.widgets.aspect_ratio_widget import AspectRatioWidget
from ui.widgets.log_viewer import LogViewer
from core.constants import DATASET_PATH, CLASSIFIER_MODEL_PATH
from ui.style import ACCENT, ACCENT2, BG1, BG2, BG3, TEXT0, TEXT1, TEXT2, DANGER


def build_model_maker_tab(window) -> qtw.QWidget:
    tab = qtw.QWidget()
    root = qtw.QGridLayout(tab)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(10)

    # ── Top toolbar ────────────────────────────────────────────────────────
    toolbar = qtw.QHBoxLayout()

    window.startCaptureBtn = qtw.QPushButton("⏺  Start Capture")
    window.startCaptureBtn.setCheckable(True)
    window.startCaptureBtn.setObjectName("primaryBtn")
    window.startCaptureBtn.clicked.connect(window.startCapture)
    toolbar.addWidget(window.startCaptureBtn)

    refresh_btn = qtw.QPushButton("↻  Refresh")
    refresh_btn.clicked.connect(window.refreshGestures)
    toolbar.addWidget(refresh_btn)

    window.visualizeModelBtn = qtw.QPushButton("📊  Visualize")
    window.visualizeModelBtn.clicked.connect(window.visualizeModel)
    toolbar.addWidget(window.visualizeModelBtn)

    window.trainExportModelBtn = qtw.QPushButton("⚙  Train Model")
    window.trainExportModelBtn.setObjectName("primaryBtn")
    window.trainExportModelBtn.clicked.connect(window.trainExportModel)
    toolbar.addWidget(window.trainExportModelBtn)

    window.reloadModelBtn = qtw.QPushButton("↺  Reload Model")
    window.reloadModelBtn.clicked.connect(window.reloadGestureModel)
    toolbar.addWidget(window.reloadModelBtn)

    open_folder_btn = qtw.QPushButton("📁  Open Folder")
    open_folder_btn.clicked.connect(window.openVersionFolder)
    toolbar.addWidget(open_folder_btn)

    toolbar.addStretch()

    window.quitProgramBtn = qtw.QPushButton("✕  Quit")
    window.quitProgramBtn.setObjectName("dangerBtn")
    window.quitProgramBtn.clicked.connect(window.close)
    toolbar.addWidget(window.quitProgramBtn)

    root.addLayout(toolbar, 0, 0, 1, -1)

    # ── Left: gesture management ───────────────────────────────────────────
    left = qtw.QVBoxLayout()
    left.setSpacing(6)

    left.addWidget(_section_label("GESTURE MANAGEMENT"))

    # Add gesture row
    add_row = qtw.QHBoxLayout()
    window.gestureNameInput = qtw.QLineEdit()
    window.gestureNameInput.setPlaceholderText("New gesture name…")
    window.gestureNameInput.returnPressed.connect(window.gestureNameExistsCheck)
    add_row.addWidget(window.gestureNameInput)
    add_btn = qtw.QPushButton("Add")
    add_btn.setObjectName("primaryBtn")
    add_btn.clicked.connect(window.gestureNameExistsCheck)
    add_row.addWidget(add_btn)
    del_btn = qtw.QPushButton("Delete")
    del_btn.setObjectName("dangerBtn")
    del_btn.clicked.connect(window.gestureSelectedCheck)
    add_row.addWidget(del_btn)
    left.addLayout(add_row)

    # Gesture tree
    window.listGesturesTree = qtw.QTreeWidget()
    window.listGesturesTree.setHeaderHidden(True)
    window.listGesturesTree.setRootIsDecorated(False)
    window.listGesturesTree.setDragEnabled(True)
    window.listGesturesTree.setAcceptDrops(True)
    window.listGesturesTree.setDropIndicatorShown(True)
    window.listGesturesTree.setDragDropMode(qtw.QAbstractItemView.DragDropMode.InternalMove)
    left.addWidget(window.listGesturesTree, stretch=1)

    # Gesture info tree
    left.addWidget(_section_label("IMAGE COUNTS"))
    window.gestureTreeInfo = qtw.QTreeWidget()
    window.gestureTreeInfo.setColumnCount(2)
    window.gestureTreeInfo.setHeaderLabels(["Gesture", "Images"])
    window.gestureTreeInfo.header().setSectionResizeMode(qtw.QHeaderView.ResizeMode.Stretch)
    window.gestureTreeInfo.setMaximumHeight(160)
    left.addWidget(window.gestureTreeInfo)

    left_widget = qtw.QWidget()
    left_widget.setLayout(left)
    left_widget.setFixedWidth(280)
    root.addWidget(left_widget, 1, 0)

    # ── Center: camera view ────────────────────────────────────────────────
    center = qtw.QVBoxLayout()
    center.addWidget(_section_label("CAPTURE VIEW"))
    window.cameraView = AspectRatioWidget(16/9)
    window.cameraView.setMinimumSize(360, 200)
    center.addWidget(window.cameraView, stretch=1)

    # Capture status
    window.captureStatusLabel = qtw.QLabel("Select a gesture and press Start Capture")
    window.captureStatusLabel.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
    window.captureStatusLabel.setStyleSheet(f"color: {TEXT1}; font-size: 12px;")
    center.addWidget(window.captureStatusLabel)

    # Image count for selected gesture
    window.selectedGestureCountLabel = qtw.QLabel("")
    window.selectedGestureCountLabel.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
    window.selectedGestureCountLabel.setObjectName("confidenceLabel")
    center.addWidget(window.selectedGestureCountLabel)

    center_widget = qtw.QWidget()
    center_widget.setLayout(center)
    root.addWidget(center_widget, 1, 1)

    # ── Right: training status ─────────────────────────────────────────────
    right = qtw.QVBoxLayout()
    right.setSpacing(6)

    right.addWidget(_section_label("TRAINING STATUS"))

    # Training progress bar
    window.trainingProgressBar = qtw.QProgressBar()
    window.trainingProgressBar.setRange(0, 100)
    window.trainingProgressBar.setValue(0)
    window.trainingProgressBar.setTextVisible(True)
    window.trainingProgressBar.setFormat("Ready")
    window.trainingProgressBar.setFixedHeight(20)
    right.addWidget(window.trainingProgressBar)

    # Accuracy display
    acc_row = qtw.QHBoxLayout()
    acc_row.addWidget(_section_label("LAST ACCURACY"))
    window.accuracyLabel = qtw.QLabel("—")
    window.accuracyLabel.setObjectName("confidenceLabel")
    window.accuracyLabel.setAlignment(qtc.Qt.AlignmentFlag.AlignRight)
    acc_row.addWidget(window.accuracyLabel)
    right.addLayout(acc_row)

    window.accuracyBar = qtw.QProgressBar()
    window.accuracyBar.setObjectName("accuracyBar")
    window.accuracyBar.setRange(0, 100)
    window.accuracyBar.setValue(0)
    window.accuracyBar.setFixedHeight(8)
    window.accuracyBar.setTextVisible(False)
    right.addWidget(window.accuracyBar)

    # Training log
    right.addWidget(_section_label("TRAINING LOG"))
    window.statusOutput = LogViewer(maxLines=2000)
    right.addWidget(window.statusOutput, stretch=1)

    # Dataset path
    path_lbl = qtw.QLabel(f"📁 {DATASET_PATH}")
    path_lbl.setStyleSheet(f"color: {TEXT2}; font-size: 11px;")
    path_lbl.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextSelectableByMouse)
    path_lbl.setCursor(qtc.Qt.CursorShape.PointingHandCursor)
    path_lbl.mousePressEvent = lambda e: __import__('os').startfile(str(DATASET_PATH))
    right.addWidget(path_lbl)

    right_widget = qtw.QWidget()
    right_widget.setLayout(right)
    root.addWidget(right_widget, 1, 2)

    # ── Bottom: full-width training log ────────────────────────────────────
    root.setColumnStretch(0, 0)
    root.setColumnStretch(1, 1)
    root.setColumnStretch(2, 1)
    root.setRowStretch(1, 1)

    return tab


def _section_label(text: str) -> qtw.QLabel:
    lbl = qtw.QLabel(text)
    lbl.setObjectName("sectionHeader")
    return lbl