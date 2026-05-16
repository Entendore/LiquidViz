import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor, QPalette, Qt
from ui import LiquidWindow

def main():
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(60, 120, 180))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = LiquidWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()