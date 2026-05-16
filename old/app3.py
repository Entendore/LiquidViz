import sys
import numpy as np
import cv2
import math
from datetime import datetime

# PySide6 Imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QStatusBar, QFrame)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# Optional: Pillow for GIF support
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ========== CONFIGURATION ==========
SIM_W, SIM_H = 480, 360
FPS = 60

# Physics Constants
FLOW_RATE = 0.25
GRAVITY = 0.18
VISCOSITY = 0.02
MAX_WATER = 1.0

# ========== MINIMALIST PALETTES ==========
def palette_ink(v):
    i = int(v * 255)
    return (i, i, min(255, i + int(v*10))) 

def palette_obsidian(v):
    base = int(v * 200)
    return (base, base // 2, base // 3)

def palette_jade(v):
    return (int(30 + 60*v), int(80 + 140*v), int(70 + 100*v))

def palette_cobalt(v):
    return (int(20 + 50*v), int(40 + 80*v), int(120 + 135*v))

def palette_amber(v):
    return (int(200 + 55*v), int(150 + 80*v), int(20 + 40*v))

PALETTES = [
    ("Ink", palette_ink),
    ("Obsidian", palette_obsidian),
    ("Jade", palette_jade),
    ("Cobalt", palette_cobalt),
    ("Amber", palette_amber)
]

# ========== OPTIMIZED ENGINE ==========
class LiquidEngine:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.water = np.zeros((h, w), dtype=np.float32)
        self.palette_idx = 0
        self._build_lut()
        
    def _build_lut(self):
        lut = []
        for i in range(256):
            v = i / 255.0
            r, g, b = PALETTES[self.palette_idx][1](v)
            lut.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        self.lut = np.array(lut, dtype=np.uint8)
    
    def set_palette(self, idx):
        self.palette_idx = idx % len(PALETTES)
        self._build_lut()

    def seed_solid(self):
        self.water.fill(0)
        cx, cy = self.w // 2, self.h // 2
        self.water[cy-20:cy+20, cx-40:cx+40] = 0.8
        self.water[cy-30:cy, cx-10:cx+10] = 0.6

    def seed_wave(self):
        self.water.fill(0)
        x = np.arange(self.w)
        for row in range(self.h // 2, self.h):
            offset = int(20 * math.sin(x * 0.05 + row * 0.1))
            self.water[row, (x + offset) % self.w] = 0.7

    def seed_rain(self):
        self.water.fill(0)
        xs = np.random.randint(0, self.w, 50)
        ys = np.random.randint(0, self.h // 3, 50)
        for x, y in zip(xs, ys):
            self.water[y:y+5, x:x+5] = 0.6

    def update(self):
        # Gravity
        current = self.water[:-1, :]
        below = self.water[1:, :]
        diff = current - below
        flow = np.maximum(diff, 0) * FLOW_RATE
        self.water[:-1, :] -= flow
        self.water[1:, :] += flow
        
        # Dispersion
        kernel = np.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]], dtype=np.float32)
        self.water = cv2.filter2D(self.water, -1, kernel)
        
        # Damping
        self.water *= (1.0 - VISCOSITY)
        np.clip(self.water, 0, MAX_WATER, out=self.water)

    def render(self, target_size):
        indices = np.clip((self.water * 255).astype(np.uint8), 0, 255)
        rgb_small = self.lut[indices] # (H, W, 3)
        return cv2.resize(rgb_small, target_size, interpolation=cv2.INTER_LINEAR)

# ========== RECORDER ==========
class Recorder:
    def __init__(self):
        self.video_out = None
        self.gif_frames = []
        self.active = False
        self.format = 'mp4'
        
    def start(self, filename, fps, resolution, fmt='mp4'):
        self.stop()
        if fmt == 'gif' and not PIL_AVAILABLE: fmt = 'mp4'
        
        self.format = fmt
        if fmt == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_out = cv2.VideoWriter(filename, fourcc, fps, resolution)
        elif fmt == 'gif':
            self.gif_frames = []
            self.gif_name = filename
            
        self.active = True
        return True

    def add_frame(self, frame):
        if not self.active: return
        if self.format == 'mp4':
            self.video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif self.format == 'gif':
            self.gif_frames.append(PILImage.fromarray(frame))

    def stop(self):
        if not self.active: return
        if self.format == 'mp4' and self.video_out:
            self.video_out.release()
        elif self.format == 'gif' and self.gif_frames:
            self.gif_frames[0].save(self.gif_name, save_all=True, 
                                    append_images=self.gif_frames[1:], 
                                    duration=int(1000/FPS), loop=0)
        
        self.active = False
        self.video_out = None
        self.gif_frames = []

# ========== MAIN WINDOW ==========
class LiquidWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liquid Simulation • PySide6")
        self.resize(800, 600)
        
        # State
        self.sim = LiquidEngine(SIM_W, SIM_H)
        self.sim.seed_solid()
        self.recorder = Recorder()
        
        self.timer_palette = 0
        self.timer_preset = 0
        self.current_preset = 0
        self.recording = False
        
        # Setup UI
        self._setup_ui()
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(int(1000 / FPS))

    def _setup_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top Bar (Controls)
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #1e1e1e; padding: 5px;")
        bar_layout = QHBoxLayout(top_bar)
        
        self.btn_rec = QPushButton("⏵ Record MP4")
        self.btn_rec.setStyleSheet("color: white; background-color: #3d3d3d; border-radius: 4px; padding: 8px;")
        self.btn_rec.clicked.connect(self.toggle_recording)
        
        self.btn_gif = QPushButton("⏵ Record GIF")
        self.btn_gif.setStyleSheet("color: white; background-color: #3d3d3d; border-radius: 4px; padding: 8px;")
        self.btn_gif.clicked.connect(lambda: self.toggle_recording(fmt='gif'))
        if not PIL_AVAILABLE:
            self.btn_gif.setEnabled(False)
            self.btn_gif.setToolTip("Install Pillow to enable GIF export")

        bar_layout.addWidget(self.btn_rec)
        bar_layout.addWidget(self.btn_gif)
        bar_layout.addStretch()
        
        # Display Area
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: black;")
        
        # Status Bar
        self.statusBar().showMessage(f"Palette: {PALETTES[0][0]} | Press Record to export video.")
        self.statusBar().setStyleSheet("color: #888;")

        layout.addWidget(top_bar)
        layout.addWidget(self.display_label, 1)

    def toggle_recording(self, fmt='mp4'):
        if self.recording:
            self.recorder.stop()
            self.recording = False
            self.btn_rec.setText("⏵ Record MP4")
            self.btn_gif.setText("⏵ Record GIF")
            self.btn_rec.setStyleSheet("color: white; background-color: #3d3d3d; border-radius: 4px; padding: 8px;")
            self.statusBar().showMessage("Export complete.")
        else:
            # Open File Dialog
            ext = f".{fmt}"
            filter_str = f"Video (*{ext})"
            fname, _ = QFileDialog.getSaveFileName(self, "Save Video", f"liquid_{datetime.now().strftime('%H%M%S')}{ext}", filter_str)
            
            if fname:
                # Get current display size
                display_size = (self.display_label.width(), self.display_label.height())
                self.recorder.start(fname, FPS, display_size, fmt)
                self.recording = True
                
                if fmt == 'mp4':
                    self.btn_rec.setText("⏹ Stop Recording")
                    self.btn_rec.setStyleSheet("color: white; background-color: #800000; border-radius: 4px; padding: 8px;")
                else:
                    self.btn_gif.setText("⏹ Stop Recording")
                    self.btn_gif.setStyleSheet("color: white; background-color: #800000; border-radius: 4px; padding: 8px;")
                
                self.statusBar().showMessage(f"Recording to {fname}...")

    def update_loop(self):
        # 1. Physics
        dt = 1.0 / FPS
        self.timer_palette += dt
        self.timer_preset += dt
        
        # Autonomous Logic
        if self.timer_palette > 6.0:
            self.timer_palette = 0
            self.sim.set_palette(self.sim.palette_idx + 1)
            self.statusBar().showMessage(f"Palette changed to: {PALETTES[self.sim.palette_idx][0]}")
            
        if self.timer_preset > 10.0:
            self.timer_preset = 0
            self.current_preset = (self.current_preset + 1) % 3
            if self.current_preset == 0: self.sim.seed_solid()
            elif self.current_preset == 1: self.sim.seed_wave()
            else: self.sim.seed_rain()

        # Run Physics steps
        for _ in range(3):
            self.sim.update()

        # 2. Render
        w = self.display_label.width()
        h = self.display_label.height()
        if w < 1 or h < 1: return
        
        frame = self.sim.render((w, h)) # Returns RGB numpy array (H, W, 3)
        
        # 3. Convert to Qt Image
        # QImage needs (W, H) for stride calculation
        h_img, w_img, ch = frame.shape
        bytes_per_line = 3 * w_img
        q_img = QImage(frame.data, w_img, h_img, bytes_per_line, QImage.Format_RGB888)
        
        # 4. Display
        self.display_label.setPixmap(QPixmap.fromImage(q_img))
        
        # 5. Record if active
        if self.recording:
            self.recorder.add_frame(frame)

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Dark Theme defaults
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(40, 40, 40))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(60, 60, 60))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    window = LiquidWindow()
    window.show()
    sys.exit(app.exec())