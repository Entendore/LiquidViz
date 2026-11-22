import sys
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QFrame, QRadioButton)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QColor, QPalette

from config import FPS, RES_STANDARD, RES_SHORTS
from engine import LiquidEngine
from recorder import Recorder

class LiquidWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liquid Sim • Shallow Water GPU")
        self.resize(900, 700)
        
        self.sim = LiquidEngine()
        self.sim.seed_solid()
        self.recorder = Recorder()
        
        self.timer_palette = 0
        self.current_preset = 0
        self.recording = False
        self.export_resolution = RES_STANDARD
        
        self._setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(int(1000 / FPS))

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #2b2b2b; padding: 8px; border-bottom: 1px solid #444;")
        bar_layout = QHBoxLayout(top_bar)
        
        self.radio_standard = QRadioButton("YouTube (16:9)")
        self.radio_standard.setChecked(True)
        self.radio_standard.setStyleSheet("color: white; font-weight: bold;")
        self.radio_standard.toggled.connect(lambda: self.set_export_mode(RES_STANDARD))

        self.radio_shorts = QRadioButton("Shorts (9:16)")
        self.radio_shorts.setStyleSheet("color: white; font-weight: bold;")
        self.radio_shorts.toggled.connect(lambda: self.set_export_mode(RES_SHORTS))

        bar_layout.addWidget(self.radio_standard)
        bar_layout.addWidget(self.radio_shorts)
        
        bar_layout.addSpacing(20)
        bar_layout.addWidget(self._vertical_line())
        bar_layout.addSpacing(20)

        self.btn_rec = QPushButton("⏵ Record MP4")
        self.btn_rec.setStyleSheet(self._btn_style("#3d3d3d"))
        self.btn_rec.clicked.connect(self.toggle_recording)

        bar_layout.addWidget(self.btn_rec)
        bar_layout.addStretch()
        
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #000;")
        
        self.statusBar().showMessage(f"Ready. Output: {RES_STANDARD[0]}x{RES_STANDARD[1]}")
        self.statusBar().setStyleSheet("color: #aaa; background-color: #222;")

        layout.addWidget(top_bar)
        layout.addWidget(self.display_label, 1)

    def _btn_style(self, bg_color):
        return f"""
            QPushButton {{
                color: white; background-color: {bg_color}; 
                border-radius: 4px; padding: 8px 16px;
                font-weight: bold; border: 1px solid #555;
            }}
            QPushButton:hover {{ background-color: #555; }}
            QPushButton:disabled {{ background-color: #222; color: #555; }}
        """

    def _vertical_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: #555;")
        return line

    def set_export_mode(self, res):
        self.export_resolution = res
        w, h = res
        mode = "YouTube" if w > h else "Shorts"
        self.statusBar().showMessage(f"Output: {mode} ({w}x{h})")

    def toggle_recording(self):
        if self.recording:
            self.recorder.stop()
            self.recording = False
            self.btn_rec.setText("⏵ Record MP4")
            self.btn_rec.setStyleSheet(self._btn_style("#3d3d3d"))
            self.statusBar().showMessage("Export complete.")
            self.radio_standard.setEnabled(True)
            self.radio_shorts.setEnabled(True)
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode = "shorts" if self.export_resolution[0] < self.export_resolution[1] else "video"
            default_name = f"liquid_{mode}_{ts}.mp4"
            
            fname, _ = QFileDialog.getSaveFileName(self, "Save Video", default_name, "Video (*.mp4)")
            
            if fname:
                self.radio_standard.setEnabled(False)
                self.radio_shorts.setEnabled(False)
                
                if self.recorder.start(fname, self.export_resolution):
                    self.recording = True
                    self.btn_rec.setText("⏹ Stop Recording")
                    self.btn_rec.setStyleSheet(self._btn_style("#800000"))
                    w, h = self.export_resolution
                    self.statusBar().showMessage(f"Recording {w}x{h}...")
                else:
                    self.statusBar().showMessage("Error: FFmpeg not found.")

    def update_loop(self):
        # 1. Physics Loop
        self.sim.update()
        
        # 2. Logic: Cycle Palettes and Effects
        dt = 1.0 / FPS
        self.timer_palette += dt
        
        if self.timer_palette > 8.0:
            self.timer_palette = 0
            self.sim.set_palette(self.sim.palette_idx + 1)
            
        # 3. Render
        frame = self.sim.render() # Returns numpy RGB
        
        # 4. Record
        if self.recording:
            self.recorder.add_frame(frame)
        
        # 5. Display
        h, w, ch = frame.shape
        bytes_per_line = 3 * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        lbl_w = self.display_label.width()
        lbl_h = self.display_label.height()
        pixmap = QPixmap.fromImage(q_img).scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(pixmap)