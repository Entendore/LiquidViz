import sys
import time
import numpy as np
import colorsys
import random
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QComboBox, QPushButton, QLabel, QFileDialog)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPainter, QColor, QFont

from config import PRESETS, RENDER_MODES, SIM_GRID_SIZE, FPS
from engine import FluidEngine
from recorder import VideoRecorder

class SimCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = None
        self.fps_str = "0 FPS"
        self.status_str = "Ready"

    def update_image(self, img_array):
        self._img = img_array
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        if self._img is not None:
            h, w, c = self._img.shape
            qimg = QImage(self._img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            p.drawImage(self.rect(), qimg)
        else:
            p.fillRect(self.rect(), QColor(0, 0, 0))

        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Consolas", 10))
        p.drawText(10, 20, self.fps_str)
        p.drawText(10, 40, self.status_str)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidViz - Full Color")
        self.resize(900, 900)

        self.engine = FluidEngine(SIM_GRID_SIZE)
        self.recorder = VideoRecorder()
        self.time = 0.0
        
        self.current_preset = list(PRESETS.keys())[0]
        self.render_mode = 0
        self.is_paused = False
        self.is_recording = False
        
        self.frame_times = []
        
        # Initialize seeds before warmup
        self.seeds = [random.random() * 100 for _ in range(20)]

        self.setup_ui()
        
        # --- WARMUP PHASE ---
        # Run the simulation in background to fill the screen before showing window
        print("Initializing fluid simulation...")
        self.warm_up_simulation(steps=200) 
        print("Ready.")
        
        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(int(1000 / FPS))

    def warm_up_simulation(self, steps=200):
        """Runs the simulation engine rapidly to pre-fill the screen."""
        # Temporarily speed up time for injection
        original_time = self.time
        
        for _ in range(steps):
            # Advance time faster for warmup to spread colors quickly
            self.time += (1.0 / FPS) * 2.0 
            self.inject_autonomous()
            self.engine.step(0.025)
            
        # Render one frame to have something ready for the canvas
        frame = self.engine.render(self.time, self.render_mode)
        self.canvas.update_image(frame)

    def setup_ui(self):
        lay = QVBoxLayout(self)
        h = QHBoxLayout()
        
        h.addWidget(QLabel("Preset:"))
        self.combo = QComboBox()
        self.combo.addItems(PRESETS.keys())
        self.combo.currentTextChanged.connect(self.change_preset)
        h.addWidget(self.combo)
        
        h.addWidget(QLabel("Visual:"))
        self.combo_vis = QComboBox()
        self.combo_vis.addItems(RENDER_MODES)
        self.combo_vis.currentTextChanged.connect(lambda t: setattr(self, 'render_mode', RENDER_MODES.index(t)))
        h.addWidget(self.combo_vis)
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        h.addWidget(self.btn_pause)
        
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_sim)
        h.addWidget(self.btn_reset)
        
        self.btn_rec = QPushButton("Start Record")
        self.btn_rec.clicked.connect(self.toggle_record)
        h.addWidget(self.btn_rec)
        
        h.addStretch()
        lay.addLayout(h)
        
        self.canvas = SimCanvas()
        lay.addWidget(self.canvas, 1)

    def change_preset(self, name):
        self.current_preset = name
        self.engine.reset()
        self.seeds = [random.random() * 100 for _ in range(20)]
        # Run a mini-warmup on preset change so user doesn't see black screen
        self.warm_up_simulation(steps=100) 

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.setText("Resume" if self.is_paused else "Pause")

    def reset_sim(self):
        self.engine.reset()
        self.seeds = [random.random() * 100 for _ in range(20)]
        self.warm_up_simulation(steps=100)

    def toggle_record(self):
        if not self.is_recording:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Video", "output.mp4", "*.mp4")
            if filename:
                if self.recorder.start(filename, FPS, (SIM_GRID_SIZE, SIM_GRID_SIZE), (1280, 720)):
                    self.is_recording = True
                    self.btn_rec.setText("Stop Record")
        else:
            self.recorder.stop()
            self.is_recording = False
            self.btn_rec.setText("Start Record")

    def inject_autonomous(self):
        p = PRESETS[self.current_preset]
        N = SIM_GRID_SIZE
        colors_config = p["colors"]
        
        num = p["num_sources"]
        
        for i in range(num):
            seed = self.seeds[i]
            t = self.time * p["speed"] + seed
            
            # Use chaotic frequencies to cover the screen
            freq_x = 2.0 + (i % 3) * 0.7
            freq_y = 3.0 + (i % 4) * 0.6
            phase = seed * 6.28
            
            radius = p["radius"]
            cx = 0.5 + radius * np.sin(t * freq_x + phase)
            cy = 0.5 + radius * np.cos(t * freq_y)
            
            gx = int(cx * N)
            gy = int(cy * N)
            
            if 1 <= gx <= N and 1 <= gy <= N:
                vx = np.cos(t * freq_x + phase) * freq_x * radius * p["force"]
                vy = -np.sin(t * freq_y) * freq_y * radius * p["force"]
                
                # Turbulence
                turb = np.sin(t * 50) * 2.0
                vx += turb
                vy += turb
                
                if colors_config == "rainbow":
                    hue = (self.time * 0.1 + i * 0.1) % 1.0
                    color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                else:
                    c_idx = int(self.time * 0.5 + i) % len(colors_config)
                    color = colors_config[c_idx]
                
                self.engine.add_source(gx, gy, p["density"], float(p["brush_size"]), vx, vy, color)

    def game_loop(self):
        now = time.perf_counter()
        self.frame_times.append(now)
        if len(self.frame_times) > 60: self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            dt = (self.frame_times[-1] - self.frame_times[0]) / len(self.frame_times)
            self.canvas.fps_str = f"{int(1.0/dt)} FPS"

        if not self.is_paused:
            self.time += (1.0 / FPS)
            self.inject_autonomous()
            self.engine.step(0.025)
        
        rec_status = " [REC]" if self.is_recording else ""
        pause_status = " [PAUSED]" if self.is_paused else ""
        self.canvas.status_str = f"{self.current_preset}{rec_status}{pause_status}"
        
        frame = self.engine.render(self.time, self.render_mode)
        self.canvas.update_image(frame)
        
        if self.is_recording:
            self.recorder.add_frame(frame)

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()