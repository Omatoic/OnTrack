"""Application-wide dark stylesheet for the GLOMAP tracker UI."""

DARK_STYLESHEET = """
/* ---------- Palette ----------
   bg       #121212
   surface  #1e1e1e
   card     #252525
   border   #333333
   primary  #3a7bd5
   success  #28a745
   danger   #dc3545
   text     #f0f0f0
   muted    #a0a0a0
*/

QWidget {
    font-family: "Segoe UI", "Microsoft Sans Serif", sans-serif;
    font-size: 10pt;
    color: #f0f0f0;
}

QMainWindow {
    background-color: #121212;
}

QGroupBox {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 12px;
    padding-right: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #f0f0f0;
    font-weight: bold;
}

QLabel {
    color: #f0f0f0;
    background-color: transparent;
}

QLabel#dropHint {
    color: #a0a0a0;
    font-size: 9pt;
}

QLineEdit {
    background-color: #252525;
    border: 1px solid #333333;
    border-radius: 5px;
    padding: 5px 8px;
    color: #f0f0f0;
}

QLineEdit:read-only {
    background-color: #1a1a1a;
    color: #a0a0a0;
}

QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #252525;
    border: 1px solid #333333;
    border-radius: 5px;
    padding: 4px;
    color: #f0f0f0;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #252525;
    color: #f0f0f0;
    border: 1px solid #333333;
    selection-background-color: #3a7bd5;
}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #333333;
    border-radius: 3px;
    margin: 1px;
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #444444;
}

QPushButton {
    background-color: #3a7bd5;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #4a8be5;
}

QPushButton:pressed {
    background-color: #2a6bc5;
}

QPushButton:disabled {
    background-color: #333333;
    color: #888888;
}

QPushButton#runButton {
    background-color: #28a745;
}

QPushButton#runButton:hover {
    background-color: #34ce57;
}

QPushButton#runButton:pressed {
    background-color: #1e7e34;
}

QPushButton#runButton:disabled {
    background-color: #333333;
    color: #888888;
}

QPushButton#cancelButton {
    background-color: #dc3545;
}

QPushButton#cancelButton:hover {
    background-color: #e04b59;
}

QPushButton#cancelButton:pressed {
    background-color: #b02a37;
}

QProgressBar {
    border: 1px solid #333333;
    border-radius: 5px;
    text-align: center;
    background-color: #252525;
    color: #f0f0f0;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #3a7bd5;
    border-radius: 4px;
}

QTextEdit {
    background-color: #0d0d0d;
    color: #d4d4d4;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9pt;
}

QCheckBox {
    spacing: 6px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #333333;
    background-color: #252525;
}

QCheckBox::indicator:checked {
    background-color: #3a7bd5;
}

QStatusBar {
    background-color: #1e1e1e;
    color: #a0a0a0;
    border-top: 1px solid #333333;
}

QMessageBox {
    background-color: #1e1e1e;
}

QMessageBox QLabel {
    color: #f0f0f0;
}

QMessageBox QPushButton {
    min-width: 70px;
}
"""


def apply(app) -> None:
    """Apply the dark stylesheet to a QApplication instance."""
    app.setStyleSheet(DARK_STYLESHEET)
