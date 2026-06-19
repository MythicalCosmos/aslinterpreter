import main
#if main.theme == "Dark":
ACCENT   = "#00D4AA"    # Electric teal
ACCENT2  = "#F5A623"    # Warm amber (confidence / warnings)
DANGER   = "#FF4757"    # Red (errors)
BG0      = "#0D0F14"    # Deepest background
BG1      = "#13161D"    # Card background
BG2      = "#1A1E28"    # Elevated surface
BG3      = "#222636"    # Border / separator
TEXT0    = "#E8EAF0"    # Primary text
TEXT1    = "#9BA3B8"    # Secondary text
TEXT2    = "#5C6478"    # Muted / placeholder


STYLESHEET = f"""
/* ── Base ───────────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {BG0};
    color: {TEXT0};
    font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', sans-serif;
    font-size: 13px;
}}

/* ── Tab Bar ─────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BG3};
    background-color: {BG1};
    border-radius: 8px;
}}
QTabBar::tab {{
    background: {BG2};
    color: {TEXT1};
    padding: 10px 24px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.5px;
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
    background: {BG1};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT0};
    background: {BG3};
}}

/* ── Buttons ─────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG2};
    color: {TEXT0};
    border: 1px solid {BG3};
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {BG3};
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {BG0};
}}
QPushButton:checked {{
    background-color: {ACCENT};
    color: {BG0};
    border-color: {ACCENT};
    font-weight: 700;
}}
QPushButton:disabled {{
    color: {TEXT2};
    border-color: {BG3};
}}
QPushButton#dangerBtn {{
    border-color: {DANGER};
    color: {DANGER};
}}
QPushButton#dangerBtn:hover {{
    background-color: {DANGER};
    color: white;
}}
QPushButton#primaryBtn {{
    background-color: {ACCENT};
    color: {BG0};
    border-color: {ACCENT};
    font-weight: 700;
}}
QPushButton#primaryBtn:hover {{
    background-color: #00FFCC;
    border-color: #00FFCC;
}}

/* ── Text Displays ───────────────────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background-color: {BG1};
    color: {TEXT0};
    border: 1px solid {BG3};
    border-radius: 6px;
    padding: 8px;
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 13px;
    selection-background-color: {ACCENT};
    selection-color: {BG0};
}}

/* ── Labels ──────────────────────────────────────────────────────────── */
QLabel {{
    color: {TEXT1};
    font-size: 12px;
}}
QLabel#sectionHeader {{
    color: {ACCENT};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}
QLabel#primaryOutput {{
    color: {TEXT0};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 22px;
    font-weight: 700;
    padding: 12px;
    background-color: {BG1};
    border: 1px solid {BG3};
    border-radius: 8px;
}}
QLabel#confidenceLabel {{
    color: {ACCENT2};
    font-size: 13px;
    font-weight: 600;
}}
QLabel#statusOk {{ color: {ACCENT}; font-weight: 600; }}
QLabel#statusWarn {{ color: {ACCENT2}; font-weight: 600; }}
QLabel#statusError {{ color: {DANGER}; font-weight: 600; }}

/* ── Input Controls ──────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {BG1};
    color: {TEXT0};
    border: 1px solid {BG3};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit::placeholder {{
    color: {TEXT2};
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {BG1};
    color: {TEXT0};
    border: 1px solid {BG3};
    border-radius: 6px;
    padding: 6px 10px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}

QComboBox {{
    background-color: {BG1};
    color: {TEXT0};
    border: 1px solid {BG3};
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 120px;
}}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background-color: {BG2};
    color: {TEXT0};
    selection-background-color: {ACCENT};
    selection-color: {BG0};
    border: 1px solid {BG3};
}}

QCheckBox {{
    color: {TEXT0};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BG3};
    border-radius: 4px;
    background: {BG1};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Tree Widgets ────────────────────────────────────────────────────── */
QTreeWidget {{
    background-color: {BG1};
    color: {TEXT0};
    border: 1px solid {BG3};
    border-radius: 6px;
    alternate-background-color: {BG2};
    show-decoration-selected: 1;
}}
QTreeWidget::item {{
    padding: 6px 8px;
    border-radius: 4px;
}}
QTreeWidget::item:selected {{
    background-color: {ACCENT};
    color: {BG0};
}}
QTreeWidget::item:hover:!selected {{
    background-color: {BG3};
}}
QHeaderView::section {{
    background-color: {BG2};
    color: {TEXT1};
    padding: 6px 10px;
    border: none;
    border-bottom: 1px solid {BG3};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

/* ── Progress Bar ────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG2};
    border: 1px solid {BG3};
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}
QProgressBar#accuracyBar::chunk {{
    background-color: {ACCENT2};
}}

/* ── Scrollbars ──────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG1};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BG3};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT2};
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

/* ── Splitter ────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {BG3};
    width: 1px;
    height: 1px;
}}

/* ── Message Boxes ───────────────────────────────────────────────────── */
QMessageBox {{
    background-color: {BG1};
    color: {TEXT0};
}}
QMessageBox QPushButton {{
    min-width: 80px;
}}

/* ── Tooltips ────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG2};
    color: {TEXT0};
    border: 1px solid {BG3};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}
"""
