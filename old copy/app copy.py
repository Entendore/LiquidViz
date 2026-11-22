import sys
import time as time_module
import numpy as np
import subprocess
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QMessageBox, QLabel
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QImage, QColor, QPalette, QFont

# ---------------------------------------------------------------------------
#  Vectorised HSV → RGB
# ---------------------------------------------------------------------------
_R_MAP = np.array([0, 1, 2, 2, 3, 0], dtype=np.int32)
_G_MAP = np.array([3, 0, 0, 1, 2, 2], dtype=np.int32)
_B_MAP = np.array([2, 2, 3, 0, 0, 1], dtype=np.int32)

def hsv_to_rgb_vec(h, s, v, stack, out_rgb, rows, cols):
    np.clip(h, 0.0, 1.0, out=h)
    np.clip(s, 0.0, 1.0, out=s)
    np.clip(v, 0.0, 1.0, out=v)

    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    stack[0] = v
    np.subtract(1.0, s, out=stack[2])
    np.multiply(v, stack[2], out=stack[2]) 
    np.multiply(s, f, out=stack[1])
    np.subtract(1.0, stack[1], out=stack[1])
    np.multiply(v, stack[1], out=stack[1]) 
    np.subtract(1.0, f, out=stack[3])
    np.multiply(s, stack[3], out=stack[3])
    np.subtract(1.0, stack[3], out=stack[3])
    np.multiply(v, stack[3], out=stack[3]) 
    
    out_rgb[:, :, 0] = stack[_R_MAP[i], rows, cols]
    out_rgb[:, :, 1] = stack[_G_MAP[i], rows, cols]
    out_rgb[:, :, 2] = stack[_B_MAP[i], rows, cols]

# ---------------------------------------------------------------------------
#  Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "Neon Flow":        {"num": 6, "speed": 0.55, "rad": 0.18, "hue_off": 0.0,
                         "force": 2.5, "burst_thresh": 0.6, "density": 40.0, 
                         "psyche": 0.6, "grav": 0.6},
    "Chromatic Storm":  {"num": 8, "speed": 0.9, "rad": 0.08, "hue_off": 0.33,
                         "force": 3.5, "burst_thresh": 0.9, "density": 30.0, 
                         "psyche": 0.8, "grav": 1.2},
    "Deep Liquid":      {"num": 4, "speed": 0.2, "rad": 0.25, "hue_off": 0.66,
                         "force": 1.5, "burst_thresh": 0.3, "density": 60.0, 
                         "psyche": 0.3, "grav": 0.2},
    "Entropy Max":      {"num": 10,"speed": 1.2, "rad": 0.05, "hue_off": 0.0,
                         "force": 4.0, "burst_thresh": 0.99,"density": 25.0, 
                         "psyche": 1.0, "grav": 2.0},
}

RENDER_MODES = ["Flow Mapping", "Velocity Field", "Raw Density"]
VISUAL_STYLES = ["Smooth (Blur)", "Pixelated (Crisp)"]

RESOLUTIONS = {
    "Performance (256)": (256, None),
    "High Quality (512)": (512, None),
    "YouTube (1080p)": (512, (1920, 1080)),
    "Shorts (1080x1920)": (512, (1080, 1920))
}

# ---------------------------------------------------------------------------
#  Fluid Solver
# ---------------------------------------------------------------------------
class FluidSolver:
    def __init__(self, N, dt=0.1, diffusion=0.00008, viscosity=0.00003):
        self.N = N
        self.dt = 0.1 * (256.0 / N)
        self.diff = diffusion * (256.0 / N)
        self.visc = viscosity * (256.0 / N)
        
        self.sor_omega = 1.85
        self.iters = 6
        
        s = (N + 2, N + 2)
        s3 = (N + 2, N + 2, 3)
        
        self.u = np.zeros(s, np.float32)
        self.v = np.zeros(s, np.float32)
        self.d = np.zeros(s3, np.float32)
        self.u0 = np.zeros(s, np.float32)
        self.v0 = np.zeros(s, np.float32)
        self.d0 = np.zeros(s3, np.float32)
        
        self._tmp2a = np.zeros(s, np.float32)
        self._tmp2b = np.zeros(s, np.float32)
        self._tmp2c = np.zeros(s, np.float32)
        self._tmp3  = np.zeros(s3, np.float32)
        self._p     = np.zeros(s, np.float32)
        self._div   = np.zeros(s, np.float32)
        
        ii, jj = np.meshgrid(
            np.arange(1, N+1, dtype=np.float32),
            np.arange(1, N+1, dtype=np.float32), 
            indexing="ij"
        )
        self._adv_ii = ii
        self._adv_jj = jj
        self._adv_dtN = self.dt * N
        
        rad = 5
        di = np.arange(-rad, rad + 1, dtype=np.int32)
        dj = np.arange(-rad, rad + 1, dtype=np.int32)
        DI, DJ = np.meshgrid(di, dj, indexing="ij")
        dist = np.sqrt(DI.astype(np.float32)**2 + DJ.astype(np.float32)**2)
        f = np.maximum(0.0, 1.0 - dist / (rad + 0.5))
        mask = f > 0
        self._sp_di = DI[mask]
        self._sp_dj = DJ[mask]
        self._sp_f = f[mask].astype(np.float32)

    def _bnd(self, b, x):
        N = self.N
        if x.ndim == 3:
            x[0,1:-1] = x[1,1:-1]
            x[-1,1:-1] = x[-2,1:-1]
            x[1:-1,0] = x[1:-1,1]
            x[1:-1,-1] = x[1:-1,-2]
        else:
            x[0,1:-1] = -x[1,1:-1] if b==1 else x[1,1:-1]
            x[-1,1:-1] = -x[-2,1:-1] if b==1 else x[-2,1:-1]
            x[1:-1,0] = -x[1:-1,1] if b==2 else x[1:-1,1]
            x[1:-1,-1] = -x[1:-1,-2] if b==2 else x[1:-1,-2]
        
        x[0,0] = .5*(x[1,0]+x[0,1])
        x[0,-1] = .5*(x[1,-1]+x[0,-2])
        x[-1,0] = .5*(x[-2,0]+x[-1,1])
        x[-1,-1] = .5*(x[-2,-1]+x[-1,-2])

    def _lin_solve(self, b, x, x0, a, tmp):
        c = 1.0 + 4.0 * a
        c_inv = 1.0 / c
        w = self.sor_omega
        sl = (slice(1,-1), slice(1,-1))
        
        for _ in range(self.iters):
            np.add(x[:-2, 1:-1], x[2:, 1:-1], out=tmp[sl])
            np.add(tmp[sl], x[1:-1, :-2], out=tmp[sl])
            np.add(tmp[sl], x[1:-1, 2:], out=tmp[sl])
            
            np.multiply(tmp[sl], a, out=tmp[sl])
            np.add(tmp[sl], x0[sl], out=tmp[sl])
            np.multiply(tmp[sl], c_inv, out=tmp[sl])
            
            np.subtract(tmp[sl], x[sl], out=tmp[sl])
            np.multiply(tmp[sl], w, out=tmp[sl])
            np.add(x[sl], tmp[sl], out=x[sl])
            
            self._bnd(b, x)

    def _diffuse(self, b, x, x0, diff, tmp):
        a = self.dt * diff * self.N * self.N
        if a < 1e-12: 
            np.copyto(x, x0)
            return
        self._lin_solve(b, x, x0, a, tmp)

    def _advect(self, b, d, d0, u, v):
        N = self.N
        lo, hi = 0.5, N + 0.5
        
        np.nan_to_num(u, copy=False)
        np.nan_to_num(v, copy=False)
        
        x = np.clip(self._adv_ii - self._adv_dtN * u[1:-1,1:-1], lo, hi)
        y = np.clip(self._adv_jj - self._adv_dtN * v[1:-1,1:-1], lo, hi)
        
        i0 = np.floor(x).astype(np.int32)
        j0 = np.floor(y).astype(np.int32)
        i1 = i0 + 1
        j1 = j0 + 1
        
        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1
        
        i0 = np.clip(i0, 0, N)
        j0 = np.clip(j0, 0, N)
        i1 = np.clip(i1, 0, N+1)
        j1 = np.clip(j1, 0, N+1)

        if d.ndim == 2:
            d[1:-1,1:-1] = s0*(t0*d0[i0,j0] + t1*d0[i0,j1]) + s1*(t0*d0[i1,j0] + t1*d0[i1,j1])
            self._bnd(b, d)
        else:
            s0e = s0[...,None]
            t0e = t0[...,None]
            s1e = s1[...,None]
            t1e = t1[...,None]
            d[1:-1,1:-1] = s0e*(t0e*d0[i0,j0] + t1e*d0[i0,j1]) + s1e*(t0e*d0[i1,j0] + t1e*d0[i1,j1])
            self._bnd(0, d)

    def _project(self, u, v):
        N = self.N
        h = 1.0 / N
        sl = (slice(1,-1), slice(1,-1))
        div, p, tmp = self._div, self._p, self._tmp2a
        div[:]=0
        p[:]=0
        
        div[sl] = -0.5*h*(u[2:,1:-1] - u[:-2,1:-1] + v[1:-1,2:] - v[1:-1,:-2])
        self._bnd(0, div)
        
        self._lin_solve(0, p, div, 1.0, tmp)
        
        u[sl] -= 0.5*N*(p[2:,1:-1] - p[:-2,1:-1])
        v[sl] -= 0.5*N*(p[1:-1,2:] - p[1:-1,:-2])
        self._bnd(1,u)
        self._bnd(2,v)

    def _vorticity_confinement(self, strength=8.0):
        N = self.N
        h = 1.0 / N
        inv2 = 0.5 / h
        sl = (slice(1,-1), slice(1,-1))
        
        w = self._tmp2c
        gx = self._tmp2a
        gy = self._tmp2b
        length = self._div
        
        w[sl] = ((self.v[2:,1:-1] - self.v[:-2,1:-1]) - (self.u[1:-1,2:] - self.u[1:-1,:-2])) * inv2
        ac = np.abs(w)
        
        gx[sl] = (ac[2:,1:-1] - ac[:-2,1:-1]) * inv2
        gy[sl] = (ac[1:-1,2:] - ac[1:-1,:-2]) * inv2
        
        np.multiply(gx, gx, out=length)
        np.multiply(gy, gy, out=gy)
        np.add(length, gy, out=length)
        np.sqrt(length, out=length)
        length[sl] += 1e-5
        
        gx[sl] /= length[sl]
        gy[sl] /= length[sl]
        
        sh = strength * h
        self.u[sl] += sh * gy[sl] * w[sl]
        self.v[sl] -= sh * gx[sl] * w[sl]

    def add_density(self, cx, cy, amount, color):
        N = self.N
        ni = cx + self._sp_di
        nj = cy + self._sp_dj
        ok = (ni>=1) & (ni<=N) & (nj>=1) & (nj<=N)
        ni = ni[ok]
        nj = nj[ok]
        fv = self._sp_f[ok]
        c_arr = np.array(color, dtype=np.float32)
        self.d0[ni,nj] += (np.float32(amount) * fv)[:,None] * c_arr[None,:]

    def add_velocity(self, cx, cy, vx, vy):
        N = self.N
        ni = cx + self._sp_di
        nj = cy + self._sp_dj
        ok = (ni>=1) & (ni<=N) & (nj>=1) & (nj<=N)
        ni = ni[ok]
        nj = nj[ok]
        fv = self._sp_f[ok]
        self.u0[ni,nj] += np.float32(vx) * fv
        self.v0[ni,nj] += np.float32(vy) * fv

    def apply_gravity(self, gx, gy):
        sl = (slice(1,-1), slice(1,-1))
        self.u0[sl] += np.float32(gx)
        self.v0[sl] += np.float32(gy)

    def step(self):
        self.u += self.dt * self.u0
        self.v += self.dt * self.v0
        
        np.clip(self.u, -5.0, 5.0, out=self.u)
        np.clip(self.v, -5.0, 5.0, out=self.v)
        
        self.u, self.u0 = self.u0, self.u
        self._diffuse(1, self.u, self.u0, self.visc, self._tmp2a)
        self.v, self.v0 = self.v0, self.v
        self._diffuse(2, self.v, self.v0, self.visc, self._tmp2a)
        self._project(self.u, self.v)
        
        self.u0[:] = self.u
        self.v0[:] = self.v
        self._advect(1, self.u, self.u0, self.u0, self.v0)
        self._advect(2, self.v, self.v0, self.u0, self.v0)
        self._project(self.u, self.v)
        
        self._vorticity_confinement()
        
        self.u0[:] = 0
        self.v0[:] = 0
        
        self.d += self.dt * self.d0
        self.d, self.d0 = self.d0, self.d
        self._diffuse(0, self.d, self.d0, self.diff, self._tmp3)
        self.d0[:] = self.d
        self._advect(0, self.d, self.d0, self.u, self.v)
        
        self.d0[:] = 0
        self.d *= 0.997
        
        np.nan_to_num(self.d, copy=False)
        np.nan_to_num(self.u, copy=False)
        np.nan_to_num(self.v, copy=False)

    def reset(self):
        for a in (self.u, self.v, self.d, self.u0, self.v0, self.d0):
            a[:] = 0

# ---------------------------------------------------------------------------
#  HDR Flow Renderer (Fixed for Colorful Motion)
# ---------------------------------------------------------------------------
class PsychedelicRenderer:
    def __init__(self, N):
        self.N = N
        self._rows = np.arange(N, dtype=np.int32)[:,None]
        self._cols = np.arange(N, dtype=np.int32)[None,:]
        
        self._hue  = np.zeros((N,N), np.float32)
        self._sat  = np.zeros((N,N), np.float32)
        self._val  = np.zeros((N,N), np.float32)
        self._tmp  = np.zeros((N,N), np.float32)
        self._stk  = np.zeros((4,N,N), np.float32)
        self._rgb  = np.zeros((N,N,3), np.float32)
        self._u8   = np.zeros((N,N,3), np.uint8)
        
        y, x = np.mgrid[0:N, 0:N]
        dist = np.sqrt((x - N/2.0)**2 + (y - N/2.0)**2) / (N/2.0)
        self._vig = np.clip(1.0 - dist * 0.7, 0.2, 1.0).astype(np.float32)[:, :, None]

    def process(self, density, u, v, t, psyche, mode):
        out, N = self._rgb, self.N
        
        np.nan_to_num(density, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
        np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        if mode == "Velocity Field":
            self._render_velocity(u, v)
        elif mode == "Raw Density":
            self._render_density(density)
        else:
            self._render_flow(density, u, v, t, psyche)

        # Tonemapping
        np.divide(out, (1.0 + out), out=out)
        out *= self._vig

        # Chromatic aberration
        shift = int(3 + 2 * np.sin(t * 0.5))
        if shift > 0:
            out[:-shift, :, 0] = out[shift:, :, 0]
            out[shift:, :, 2]  = out[:-shift, :, 2]

        np.clip(out, 0.0, 1.0, out=out)
        np.sqrt(out, out=out) 
        np.multiply(out, 255.0, out=out)
        self._u8[:] = out
        
        return self._u8

    def _render_flow(self, density, u, v, t, psyche):
        N = self.N
        
        # 1. Calculate Speed and Angle
        speed = np.sqrt(u*u + v*v) + 1e-5
        angle = np.arctan2(v, u)
        
        # 2. HUE: Based on Angle (Direction) + Time
        # This creates the swirling rainbow liquid effect
        hue = (angle / (2.0 * np.pi)) + 0.5
        
        # Shift hue over time so the colors evolve
        hue += t * 0.05 
        
        # Add influence from density color if desired (subtle)
        hue += density[:,:,0] * 0.05
        hue %= 1.0
        
        # 3. SATURATION: High saturation everywhere for vivid colors
        # Areas with high speed are slightly more saturated
        sat = np.clip(speed * 4.0 + 0.5, 0.6, 1.0)
        
        # 4. VALUE (BRIGHTNESS): The key to removing the "black screen"
        # Brightness = Density Dye + Velocity Glow
        # We add 'speed' to brightness so the moving fluid glows even if dye is low
        val = np.sum(density, axis=-1, out=self._val)
        
        # Add a glow based on velocity magnitude. 
        # This ensures that movement itself is visible and colorful.
        val += speed * 1.5 
        
        # Psychedelic pulsing brightness
        if psyche > 0.1:
            val += np.sin(t * 5.0) * 0.05 * psyche
            
        np.clip(val, 0.0, 1.5, out=val) # Allow >1.0 for HDR bloom effect

        hsv_to_rgb_vec(hue, sat, val, self._stk, self._rgb, self._rows, self._cols)
        
        # Extra glow for bright spots
        mask = (val > 1.0)
        self._rgb[mask] += 0.2

    def _render_velocity(self, u, v):
        self._rgb[:,:,0] = np.clip(u * 5.0 + 0.5, 0, 1)
        self._rgb[:,:,1] = np.clip(v * 5.0 + 0.5, 0, 1)
        self._rgb[:,:,2] = np.clip((u + v) * 2.5 + 0.5, 0, 1)
        self._rgb[:] += 0.1

    def _render_density(self, density):
        val = self._val
        np.sum(density, axis=-1, out=val)
        np.clip(val, 0.0, 1.0, out=val)
        
        self._rgb[:,:,0] = val * 0.8 + 0.1
        self._rgb[:,:,1] = val * 0.9 + 0.1
        self._rgb[:,:,2] = val        + 0.1

# ---------------------------------------------------------------------------
#  UI Canvas
# ---------------------------------------------------------------------------
class SimCanvas(QWidget):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.setMinimumSize(640, 640)
        self._buf = None
        self.fps_text = "0 FPS"
        self.smooth_mode = True
        
    def set_frame(self, u8, smooth):
        self._buf = u8
        self.smooth_mode = smooth
        self.update()
        
    def paintEvent(self, e):
        if self._buf is None: return
        img = QImage(self._buf.data, self.N, self.N, self.N*3, QImage.Format.Format_RGB888)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.smooth_mode)
        p.fillRect(self.rect(), QColor(10, 10, 30))
        p.drawImage(self.rect(), img)
        p.setPen(QColor(255, 255, 255, 180))
        p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        p.drawText(10, 25, self.fps_text)
        p.end()

# ---------------------------------------------------------------------------
#  Main Window
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fluid Engine — Hypnotic Edition")
        self.resize(850, 900)
        
        self.current_res_key = "High Quality (512)"
        self.GRID = 512
        self.export_res = None
        
        self.solver = FluidSolver(self.GRID)
        self.renderer = PsychedelicRenderer(self.GRID)
        self.time = 0.0
        self.current_preset = list(PRESETS.keys())[0]
        self.render_mode = RENDER_MODES[0]
        self.visual_style = VISUAL_STYLES[0]
        
        self.ffmpeg_proc = None
        self.is_recording = False
        self.record_frames_left = 0
        self.frame_times = [] 

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10,10,10,10)
        self.canvas = SimCanvas(self.GRID)
        lay.addWidget(self.canvas, stretch=1)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(PRESETS.keys())
        self.combo_preset.currentTextChanged.connect(self._on_preset)
        row1.addWidget(self.combo_preset)
        
        row1.addWidget(QLabel("Visualizer:"))
        self.combo_render = QComboBox()
        self.combo_render.addItems(RENDER_MODES)
        self.combo_render.currentTextChanged.connect(self._on_render)
        row1.addWidget(self.combo_render)
        lay.addLayout(row1)

        row1b = QHBoxLayout()
        row1b.addWidget(QLabel("Style:"))
        self.combo_style = QComboBox()
        self.combo_style.addItems(VISUAL_STYLES)
        self.combo_style.currentTextChanged.connect(self._on_style)
        row1b.addWidget(self.combo_style)
        lay.addLayout(row1b)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Export Res:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(RESOLUTIONS.keys())
        self.combo_res.setCurrentText(self.current_res_key)
        self.combo_res.currentTextChanged.connect(self._on_res)
        row2.addWidget(self.combo_res)

        self.btn_rec10 = QPushButton("⏺ Rec 10s")
        self.btn_rec10.setFixedHeight(35)
        self.btn_rec10.clicked.connect(self._start_auto_rec)
        row2.addWidget(self.btn_rec10)
        
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setFixedHeight(35)
        self.btn_stop.clicked.connect(self._force_stop)
        row2.addWidget(self.btn_stop)
        
        self.btn_rst = QPushButton("🔄 Reset")
        self.btn_rst.setFixedHeight(35)
        self.btn_rst.clicked.connect(self._reset)
        row2.addWidget(self.btn_rst)
        lay.addLayout(row2)

    def _on_preset(self, t):
        self.current_preset = t
        self._reset()

    def _on_render(self, t):
        self.render_mode = t

    def _on_style(self, t):
        self.visual_style = t

    def _reset(self):
        self.solver.reset()
        self.time = 0.0

    def _on_res(self, text):
        if self.is_recording: return
        self.current_res_key = text
        sim_n, exp_res = RESOLUTIONS[text]
        
        if sim_n == self.GRID and self.export_res == exp_res: return
        
        self.GRID = sim_n
        self.export_res = exp_res
        
        del self.solver
        del self.renderer
        self.solver = FluidSolver(self.GRID)
        self.renderer = PsychedelicRenderer(self.GRID)
        self.canvas.N = self.GRID
        self._reset()

    def _start_auto_rec(self):
        if self.is_recording: return
        self.record_frames_left = 600 
        self._start_ffmpeg()
        
    def _force_stop(self):
        if not self.is_recording or not self.ffmpeg_proc: return
        self._stop_ffmpeg()

    def _start_ffmpeg(self):
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', 
            '-s', f'{self.GRID}x{self.GRID}', '-pix_fmt', 'rgb24', '-r', '60', 
            '-i', '-', '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '18', '-pix_fmt', 'yuv420p'
        ]
        
        output_file = 'liquid_flow.mp4'
        
        if self.export_res:
            w, h = self.export_res
            scale_filter = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
            cmd.extend(['-vf', scale_filter])
            output_file = f'liquid_{w}x{h}.mp4'
            
        cmd.append(output_file)
        
        try:
            self.ffmpeg_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.is_recording = True
            self.btn_rec10.setText("⏺ REC...")
            self.btn_rec10.setEnabled(False)
        except FileNotFoundError:
            QMessageBox.critical(self, "FFmpeg Missing", "Install FFmpeg and add to PATH.")

    def _stop_ffmpeg(self):
        self.ffmpeg_proc.stdin.close()
        self.ffmpeg_proc.wait()
        self.ffmpeg_proc = None
        self.is_recording = False
        self.btn_rec10.setText("⏺ Rec 10s")
        self.btn_rec10.setEnabled(True)
        
        w, h = (self.export_res if self.export_res else (self.GRID, self.GRID))
        QMessageBox.information(self, "Saved", f"Saved as 'liquid_{w}x{h}.mp4'")

    def _auto_inject(self):
        p = PRESETS[self.current_preset]
        N = self.GRID
        t = self.time
        
        gx = np.sin(t * 0.15) * p["grav"]
        gy = np.cos(t * 0.2) * p["grav"] * 0.8
        self.solver.apply_gravity(gx, gy)

        for i in range(p["num"]):
            ph = i * 2.39996322
            ang = t * p["speed"] + ph
            r = p["rad"] * (1.0 + 0.5 * np.sin(t*0.5 + i))
            
            cx = 0.5 + r * np.cos(ang)
            cy = 0.5 + r * np.sin(ang)
            
            ix = int(np.clip(cx*N, 1, N))
            iy = int(np.clip(cy*N, 1, N))
            
            c = np.array([
                0.5 + 0.5*np.sin(t*0.8 + i), 
                0.5 + 0.5*np.cos(t*0.9 + i), 
                0.5 + 0.5*np.sin(t*0.7 - i)
            ])

            self.solver.add_density(ix, iy, p["density"], c)
            
            vx = -np.sin(ang) * p["force"] + np.cos(t*2.1+i)*0.5
            vy =  np.cos(ang) * p["force"] + np.sin(t*1.8+i)*0.5
            self.solver.add_velocity(ix, iy, vx, vy)

        if np.sin(t * 1.5) > p["burst_thresh"]:
            bx = 0.5 + 0.2*np.sin(t*3.14)
            by = 0.5 + 0.2*np.cos(t*2.71)
            ix = int(np.clip(bx*N, 1, N))
            iy = int(np.clip(by*N, 1, N))
            self.solver.add_density(ix, iy, 100.0, [1.0, 1.0, 1.0])
            self.solver.add_velocity(ix, iy, np.sin(t*4.1)*5, np.cos(t*3.7)*5)

    def _tick(self):
        now = time_module.perf_counter()
        self.frame_times.append(now)
        if len(self.frame_times) > 60: self.frame_times.pop(0)
        if len(self.frame_times) > 1:
            avg_dt = (self.frame_times[-1] - self.frame_times[0]) / (len(self.frame_times) - 1)
            self.canvas.fps_text = f"{int(1.0 / avg_dt)} FPS | {self.GRID}x{self.GRID}"

        self.time += 0.016
        self._auto_inject()
        self.solver.step()
        
        if np.isnan(self.solver.d).any():
            self._reset()
            return

        density = self.solver.d[1:self.GRID+1, 1:self.GRID+1]
        u_vel = self.solver.u[1:self.GRID+1, 1:self.GRID+1]
        v_vel = self.solver.v[1:self.GRID+1, 1:self.GRID+1]
        psyche_val = PRESETS[self.current_preset]["psyche"]
        
        is_smooth = (self.visual_style == "Smooth (Blur)")
        frame_u8 = self.renderer.process(density, u_vel, v_vel, self.time, psyche_val, self.render_mode)

        if self.is_recording and self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.stdin.write(frame_u8.tobytes())
                self.record_frames_left -= 1
                if self.record_frames_left <= 0: self._stop_ffmpeg()
            except BrokenPipeError:
                self._force_stop()

        self.canvas.set_frame(frame_u8, is_smooth)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(15,15,15))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(210,210,210))
    pal.setColor(QPalette.ColorRole.Base, QColor(10,10,10))
    pal.setColor(QPalette.ColorRole.Button, QColor(40,40,40))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(210,210,210))
    app.setPalette(pal)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()