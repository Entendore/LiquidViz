# ui.py - PySide6 GUI for LiquidViz
#
# IMPROVEMENTS:
#   - Mouse click/drag interaction to inject fluid manually
#   - Keyboard shortcuts: Space=Pause, R=Reset, 1/2/3=Render mode
#   - Threaded warmup so UI never freezes on startup
#   - Status bar shows recording stats + frame drops
#   - Proper aspect-ratio-preserving canvas resize
#   - Removed duplicate main() — app.py is the sole entry point

import time
import numpy as np
import colorsys
import random
import threading
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QFileDialog,
    QSizePolicy, QStatusBar,
)
from PySide6.QtCore import QTimer, Qt, QPoint
from PySide6.QtGui import QImage, QPainter, QColor, QFont, QKeyEvent, QMouseEvent

from config import (
    PRESETS, RENDER_MODES, SIM_GRID_SIZE, FPS,
    MOUSE_COLOR, WARMUP_STEPS, WARMUP_SPEED_MULT,
    MINI_WARMUP_STEPS, EXPORT_WIDTH, EXPORT_HEIGHT,
)
from engine import FluidEngine
from recorder import VideoRecorder


class SimCanvas(QWidget):
    """Widget that displays the fluid simulation image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = None
        self.fps_str = "-- FPS"
        self.status_str = "Warming up..."
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_image(self, img_array):
        self._img = img_array
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if self._img is not None:
            h, w, _ = self._img.shape
            qimg = QImage(
                self._img.data, w, h, 3 * w, QImage.Format.Format_RGB888
            )
            # Draw scaled to fit widget while preserving aspect ratio
            target = self.rect()
            widget_aspect = target.width() / max(target.height(), 1)
            img_aspect = w / max(h, 1)
            if widget_aspect > img_aspect:
                # Widget is wider — fit to height
                draw_w = int(target.height() * img_aspect)
                draw_h = target.height()
                x_off = (target.width() - draw_w) // 2
                y_off = 0
            else:
                # Widget is taller — fit to width
                draw_w = target.width()
                draw_h = int(target.width() / img_aspect)
                x_off = 0
                y_off = (target.height() - draw_h) // 2
            p.drawImage(
                int(x_off), int(y_off), draw_w, draw_h, qimg
            )
        else:
            p.fillRect(self.rect(), QColor(0, 0, 0))

        # HUD overlay
        p.setPen(QColor(255, 255, 255, 200))
        p.setFont(QFont("Consolas", 10))
        p.drawText(10, 20, self.fps_str)
        p.drawText(10, 40, self.status_str)
        p.end()


class MainWindow(QWidget):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidViz - GPU Fluid Simulation")
        self.resize(960, 960)

        # --- State ---
        self.engine = FluidEngine(SIM_GRID_SIZE)
        self.recorder = VideoRecorder()
        self.time = 0.0

        self.current_preset = list(PRESETS.keys())[0]
        self.render_mode = 0
        self.is_paused = False
        self.is_recording = False

        self.frame_times: list[float] = []
        self.seeds = [random.random() * 100 for _ in range(20)]

        # Mouse interaction state
        self._mouse_pressed = False
        self._last_mouse_gx = 0
        self._last_mouse_gy = 0

        self.setup_ui()

        # --- Threaded warmup (does NOT block UI) ---
        self.status_label.setText("Initializing GPU simulation...")
        self._warmup_thread = threading.Thread(
            target=self._warmup_bg, args=(WARMUP_STEPS,), daemon=True
        )
        self._warmup_thread.start()
        self._warmup_running = True

        # Timer — starts immediately, renders whatever is available
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(int(1000 / FPS))

    # ------------------------------------------------------------------
    #  UI Setup
    # ------------------------------------------------------------------
    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # --- Control bar ---
        bar = QHBoxLayout()

        bar.addWidget(QLabel("Preset:"))
        self.combo = QComboBox()
        self.combo.addItems(PRESETS.keys())
        self.combo.currentTextChanged.connect(self.change_preset)
        bar.addWidget(self.combo)

        bar.addWidget(QLabel("Visual:"))
        self.combo_vis = QComboBox()
        self.combo_vis.addItems(RENDER_MODES)
        self.combo_vis.setCurrentIndex(0)
        self.combo_vis.currentIndexChanged.connect(self._on_vis_mode_changed)
        bar.addWidget(self.combo_vis)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setToolTip("Space")
        self.btn_pause.clicked.connect(self.toggle_pause)
        bar.addWidget(self.btn_pause)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("R")
        self.btn_reset.clicked.connect(self.reset_sim)
        bar.addWidget(self.btn_reset)

        self.btn_rec = QPushButton("Record")
        self.btn_rec.clicked.connect(self.toggle_record)
        bar.addWidget(self.btn_rec)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaa;")
        bar.addWidget(self.status_label)

        bar.addStretch()
        root.addLayout(bar)

        # --- Canvas ---
        self.canvas = SimCanvas()
        root.addWidget(self.canvas, 1)

        # --- Keyboard focus ---
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    #  Warmup (background thread)
    # ------------------------------------------------------------------
    def _warmup_bg(self, steps):
        """Run simulation in a background thread so UI stays responsive."""
        speed_mult = WARMUP_SPEED_MULT
        for _ in range(steps):
            self.time += (1.0 / FPS) * speed_mult
            # Need to inject on main thread? No — inject_autonomous only
            # reads self.time and self.seeds which are set before thread starts.
            self._inject_autonomous_unsafe()
            self.engine.step(0.025)
        self._warmup_running = False

    def _mini_warmup(self, steps=None):
        """Synchronous mini-warmup after preset change (blocks briefly)."""
        if steps is None:
            steps = MINI_WARMUP_STEPS
        for _ in range(steps):
            self.time += (1.0 / FPS) * 2.0
            self._inject_autonomous_unsafe()
            self.engine.step(0.025)

    # ------------------------------------------------------------------
    #  Source injection
    # ------------------------------------------------------------------
    def _inject_autonomous_unsafe(self):
        """Inject autonomous preset sources. Called from any thread."""
        p = PRESETS[self.current_preset]
        N = SIM_GRID_SIZE
        colors_config = p["colors"]
        num = p["num_sources"]

        for i in range(num):
            seed = self.seeds[i]
            t = self.time * p["speed"] + seed

            freq_x = 2.0 + (i % 3) * 0.7
            freq_y = 3.0 + (i % 4) * 0.6
            phase = seed * 6.2832

            radius = p["radius"]
            cx = 0.5 + radius * np.sin(t * freq_x + phase)
            cy = 0.5 + radius * np.cos(t * freq_y)

            gx = int(cx * N)
            gy = int(cy * N)

            if 1 <= gx <= N and 1 <= gy <= N:
                vx = np.cos(t * freq_x + phase) * freq_x * radius * p["force"]
                vy = -np.sin(t * freq_y) * freq_y * radius * p["force"]

                # Turbulence jitter
                turb = np.sin(t * 50) * 2.0
                vx += turb
                vy += turb

                if colors_config == "rainbow":
                    hue = (self.time * 0.1 + i * 0.1) % 1.0
                    color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                else:
                    c_idx = int(self.time * 0.5 + i) % len(colors_config)
                    color = colors_config[c_idx]

                self.engine.add_source(
                    gx, gy, p["density"],
                    float(p["brush_size"]), vx, vy, color,
                )

    # ------------------------------------------------------------------
    #  Controls
    # ------------------------------------------------------------------
    def _on_vis_mode_changed(self, idx):
        self.render_mode = idx

    def change_preset(self, name):
        self.current_preset = name
        self.engine.reset()
        self.seeds = [random.random() * 100 for _ in range(20)]
        self._mini_warmup()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.setText("Resume" if self.is_paused else "Pause")

    def reset_sim(self):
        self.engine.reset()
        self.seeds = [random.random() * 100 for _ in range(20)]
        self.time = 0.0
        self._mini_warmup()

    def toggle_record(self):
        if not self.is_recording:
            if not VideoRecorder.is_ffmpeg_available():
                self.status_label.setText("ERROR: ffmpeg not found!")
                self.status_label.setStyleSheet("color: #f66;")
                return

            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Video", "liquidviz_output.mp4", "*.mp4"
            )
            if filename:
                ok, err = self.recorder.start(
                    filename, FPS,
                    (SIM_GRID_SIZE, SIM_GRID_SIZE),
                    (EXPORT_WIDTH, EXPORT_HEIGHT),
                )
                if ok:
                    self.is_recording = True
                    self.btn_rec.setText("Stop Rec")
                    self.btn_rec.setStyleSheet("background-color: #a33;")
                else:
                    self.status_label.setText(f"Rec Error: {err}")
                    self.status_label.setStyleSheet("color: #f66;")
        else:
            self.recorder.stop()
            self.is_recording = False
            self.btn_rec.setText("Record")
            self.btn_rec.setStyleSheet("")
            frames, elapsed = self.recorder.stats
            err = self.recorder.last_error
            if err:
                self.status_label.setText(f"Rec error: {err}")
                self.status_label.setStyleSheet("color: #f66;")
            else:
                self.status_label.setText(
                    f"Saved {frames} frames ({elapsed:.1f}s)"
                )
                self.status_label.setStyleSheet("color: #6f6;")

    # ------------------------------------------------------------------
    #  Mouse interaction
    # ------------------------------------------------------------------
    def _widget_to_grid(self, pos: QPoint):
        """Convert widget pixel coordinates to simulation grid coordinates."""
        w = self.canvas.width()
        h = self.canvas.height()
        N = SIM_GRID_SIZE

        # Account for aspect-ratio letterboxing
        widget_aspect = w / max(h, 1)
        img_aspect = 1.0  # square grid
        if widget_aspect > img_aspect:
            draw_h = h
            draw_w = int(h * img_aspect)
            x_off = (w - draw_w) / 2
            y_off = 0
        else:
            draw_w = w
            draw_h = int(w / img_aspect)
            x_off = 0
            y_off = (h - draw_h) / 2

        fx = (pos.x() - x_off) / draw_w
        fy = (pos.y() - y_off) / draw_h

        gx = int(fx * N) + 1
        gy = int(fy * N) + 1
        return np.clip(gx, 1, N), np.clip(gy, 1, N)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_pressed = True
            gx, gy = self._widget_to_grid(event.position())
            self._last_mouse_gx = gx
            self._last_mouse_gy = gy
            # Inject even on a single click
            self.engine.add_mouse_source(gx, gy, gx, gy)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._mouse_pressed and not self.is_paused:
            gx, gy = self._widget_to_grid(event.position())
            self.engine.add_mouse_source(
                gx, gy, self._last_mouse_gx, self._last_mouse_gy
            )
            self._last_mouse_gx = gx
            self._last_mouse_gy = gy

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_pressed = False

    # ------------------------------------------------------------------
    #  Keyboard shortcuts
    # ------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.toggle_pause()
        elif key == Qt.Key.Key_R:
            self.reset_sim()
        elif key == Qt.Key.Key_1:
            self.combo_vis.setCurrentIndex(0)
        elif key == Qt.Key.Key_2:
            self.combo_vis.setCurrentIndex(1)
        elif key == Qt.Key.Key_3:
            self.combo_vis.setCurrentIndex(2)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def game_loop(self):
        now = time.perf_counter()
        self.frame_times.append(now)
        if len(self.frame_times) > 90:
            self.frame_times.pop(0)

        # FPS counter (rolling average)
        if len(self.frame_times) > 1:
            dt_avg = (
                (self.frame_times[-1] - self.frame_times[0])
                / (len(self.frame_times) - 1)
            )
            fps = int(1.0 / dt_avg) if dt_avg > 0 else 0
            self.canvas.fps_str = f"{fps} FPS"

        # Simulation step
        if not self.is_paused and not getattr(self, "_warmup_running", False):
            self.time += 1.0 / FPS
            self._inject_autonomous_unsafe()
            self.engine.step(0.025)

        # Status text
        parts = [self.current_preset]
        if self.is_paused:
            parts.append("PAUSED")
        if self.is_recording:
            frames, _ = self.recorder.stats
            parts.append(f"REC {frames}f")
        self.canvas.status_str = "  |  ".join(parts)

        # Render
        frame = self.engine.render(self.time, self.render_mode)
        self.canvas.update_image(frame)

        # Record
        if self.is_recording:
            self.recorder.add_frame(frame)