#!/usr/bin/env python3
# app.py - Entry point for LiquidViz
#
# This is the SOLE entry point.  (ui.py no longer defines main().)
#
# IMPROVEMENTS:
#   - Graceful dependency checking with clear error messages
#   - CUDA GPU availability detection
#   - Applies dark Fusion palette
#   - Clean single-responsibility startup

import sys


def check_dependencies():
    """Verify all required packages are installed. Exit with clear message if not."""
    missing = []

    try:
        import PySide6  # noqa: F401
    except ImportError:
        missing.append("PySide6  (pip install PySide6)")

    try:
        from numba import cuda  # noqa: F401
    except ImportError:
        missing.append("numba  (pip install numba)")

    if missing:
        print("=" * 60)
        print("  LiquidViz - Missing Dependencies")
        print("=" * 60)
        for m in missing:
            print(f"  [!] {m}")
        print()
        print("  Install all with:")
        print("    pip install PySide6 numba")
        print("=" * 60)
        sys.exit(1)

    # Check CUDA driver (but don't hard-fail — just warn)
    try:
        from numba import cuda
        _ = cuda.gpus
    except Exception:
        print("=" * 60)
        print("  WARNING: No CUDA GPU detected.")
        print("  The simulation requires an NVIDIA GPU with CUDA drivers.")
        print("  The application will still launch but will crash at runtime.")
        print("=" * 60)


def main():
    check_dependencies()

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QColor, QPalette, Qt
    from ui import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(30, 30, 30))
    pal.setColor(QPalette.ColorRole.WindowText,      Qt.GlobalColor.white)
    pal.setColor(QPalette.ColorRole.Base,            QColor(25, 25, 25))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(30, 30, 30))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(50, 50, 50))
    pal.setColor(QPalette.ColorRole.ToolTipText,     Qt.GlobalColor.white)
    pal.setColor(QPalette.ColorRole.Text,            Qt.GlobalColor.white)
    pal.setColor(QPalette.ColorRole.Button,          QColor(45, 45, 45))
    pal.setColor(QPalette.ColorRole.ButtonText,      Qt.GlobalColor.white)
    pal.setColor(QPalette.ColorRole.BrightText,      QColor(255, 50, 50))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(60, 120, 180))
    pal.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(pal)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()