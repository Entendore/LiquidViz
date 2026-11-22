#!/usr/bin/env python3
"""
LiquidViz - Pygame Fluid Simulation
=====================================
Pure Python/NumPy fluid simulation rendered with Pygame.
No GPU/CUDA required.

Run:  python liquidviz.py

Controls:
  LMB drag      - inject fluid
  RMB drag      - inject fluid with random colour
  MMB drag      - attract fluid toward cursor
  Shift+LMB     - repel fluid away from cursor
  Scroll wheel  - change brush size
  Space         - pause / resume
  R             - reset simulation
  C             - clear density only
  1-8           - switch render mode
  Tab           - cycle preset
  +/-           - change simulation speed
  B             - toggle bloom
  G             - toggle gravity / buoyancy
  M             - toggle audio tone
  F11           - toggle fullscreen
  F12           - save screenshot
  Esc           - quit
"""

import sys, os, math, time, random, colorsys, struct, io
import numpy as np
import pygame

# ======================================================================
#  CONFIG
# ======================================================================

SIM_GRID_SIZE = 192
WINDOW_SIZE   = 800
FPS_CAP       = 60
DT            = 0.025
DIFFUSION     = 0.00008
VISCOSITY     = 0.00003
DENSITY_DECAY = 0.997
VELOCITY_CAP  = 5.0
VORTICITY_STR = 8.0
PRESSURE_ITER = 10
WARMUP_STEPS  = 120
WARMUP_SPEED  = 2.0

BUOYANCY_STRENGTH = 0.12   # Upward force on "hot" (red) density
GRAVITY_STRENGTH = 0.04    # Downward force on "cold" (blue) density

MOUSE_FORCE   = 8.0
MOUSE_DENSITY = 4.0
MOUSE_COLOR   = (0.4, 0.7, 1.0)
ATTRACT_FORCE = 12.0
REPEL_FORCE   = 15.0

RENDER_MODES = [
    "Vibrant Oil",     # 0
    "Neon Glow",        # 1
    "Velocity Field",   # 2
    "Thermal",          # 3
    "Plasma",           # 4
    "Ink Drop",         # 5
    "Prismatic",        # 6
    "Smoke",            # 7
]

PRESETS = {
    "Cosmic Nebula": {
        "num_sources": 8, "speed": 0.3, "radius": 0.45,
        "force": 5.0, "density": 2.0, "brush_size": 25,
        "colors": [(0.9, 0.2, 0.5), (0.2, 0.5, 1.0), (0.2, 1.0, 0.7), (1.0, 0.5, 0.2)],
    },
    "Rainbow Chaos": {
        "num_sources": 10, "speed": 0.5, "radius": 0.5,
        "force": 6.0, "density": 2.5, "brush_size": 20,
        "colors": "rainbow",
    },
    "Toxic Swamp": {
        "num_sources": 6, "speed": 0.2, "radius": 0.35,
        "force": 3.0, "density": 3.0, "brush_size": 30,
        "colors": [(0.1, 0.8, 0.4), (0.0, 0.5, 0.2), (0.6, 0.9, 0.1), (0.2, 0.2, 0.1)],
    },
    "Fire & Ice": {
        "num_sources": 8, "speed": 0.6, "radius": 0.4,
        "force": 7.0, "density": 2.0, "brush_size": 22,
        "colors": [(1.0, 0.3, 0.0), (1.0, 0.8, 0.2), (0.2, 0.4, 1.0), (0.8, 0.8, 1.0)],
    },
    "Deep Ocean": {
        "num_sources": 7, "speed": 0.15, "radius": 0.5,
        "force": 2.5, "density": 3.5, "brush_size": 35,
        "colors": [(0.0, 0.15, 0.4), (0.0, 0.3, 0.6), (0.1, 0.5, 0.5), (0.05, 0.2, 0.3)],
    },
    "Aurora": {
        "num_sources": 9, "speed": 0.25, "radius": 0.55,
        "force": 4.0, "density": 1.8, "brush_size": 28,
        "colors": [(0.1, 0.9, 0.3), (0.0, 0.6, 0.8), (0.5, 0.1, 0.9), (0.0, 1.0, 0.6)],
    },
}

# ======================================================================
#  HSV -> RGB  (vectorised)
# ======================================================================

_R_MAP = np.array([0, 1, 2, 2, 3, 0], dtype=np.int32)
_G_MAP = np.array([3, 0, 0, 1, 2, 2], dtype=np.int32)
_B_MAP = np.array([2, 2, 3, 0, 0, 1], dtype=np.int32)


def _hsv_to_rgb_vec(h, s, v, stack, out_rgb, rows, cols):
    np.clip(h, 0.0, 1.0, out=h)
    np.clip(s, 0.0, 1.0, out=s)
    np.clip(v, 0.0, 1.0, out=v)
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    stack[0] = v
    np.subtract(1.0, s, out=stack[2]); np.multiply(v, stack[2], out=stack[2])
    np.multiply(s, f, out=stack[1]); np.subtract(1.0, stack[1], out=stack[1])
    np.multiply(v, stack[1], out=stack[1])
    np.subtract(1.0, f, out=stack[3]); np.multiply(s, stack[3], out=stack[3])
    np.subtract(1.0, stack[3], out=stack[3]); np.multiply(v, stack[3], out=stack[3])
    out_rgb[:, :, 0] = stack[_R_MAP[i], rows, cols]
    out_rgb[:, :, 1] = stack[_G_MAP[i], rows, cols]
    out_rgb[:, :, 2] = stack[_B_MAP[i], rows, cols]
    return out_rgb


# ======================================================================
#  CURL NOISE  (cheap 2-D curl for organic source motion)
# ======================================================================

def _curl_noise_2d(x, y, seed=0.0, freq=3.0):
    """Return (curl_x, curl_y) of a scalar potential at positions (x, y)."""
    def _pot(px, py):
        return (np.sin(px * freq + seed) * np.cos(py * freq * 1.3 + seed * 0.7)
                + 0.5 * np.sin(px * freq * 2.1 - py * freq * 1.7 + seed * 1.3))
    eps = 0.01
    dpdy = (_pot(x, y + eps) - _pot(x, y - eps)) / (2.0 * eps)
    dpdx = (_pot(x + eps, y) - _pot(x - eps, y)) / (2.0 * eps)
    return dpdy, -dpdx   # curl of scalar = (dP/dy, -dP/dx)


# ======================================================================
#  FLUID SOLVER  (Stable Fluids, red-black Jacobi pressure)
# ======================================================================

class FluidSolver:
    def __init__(self, N, dt=DT, diffusion=DIFFUSION, viscosity=VISCOSITY):
        self.N = N
        self.dt = dt * (256.0 / N)
        self.diff = diffusion * (256.0 / N)
        self.visc = viscosity * (256.0 / N)
        self.iters = PRESSURE_ITER

        s  = (N + 2, N + 2)
        s3 = (N + 2, N + 2, 3)
        self.u  = np.zeros(s,  np.float32)
        self.v  = np.zeros(s,  np.float32)
        self.d  = np.zeros(s3, np.float32)
        self.u0 = np.zeros(s,  np.float32)
        self.v0 = np.zeros(s,  np.float32)
        self.d0 = np.zeros(s3, np.float32)

        self._tmp2a = np.zeros(s,  np.float32)
        self._tmp2b = np.zeros(s,  np.float32)
        self._tmp2c = np.zeros(s,  np.float32)
        self._tmp3  = np.zeros(s3, np.float32)
        self._p     = np.zeros(s,  np.float32)
        self._p_new = np.zeros(s,  np.float32)
        self._div   = np.zeros(s,  np.float32)

        # Pre-compute advection grids
        ii, jj = np.meshgrid(np.arange(1, N + 1, dtype=np.float32),
                             np.arange(1, N + 1, dtype=np.float32), indexing="ij")
        self._adv_ii  = ii
        self._adv_jj  = jj
        self._adv_dtN = self.dt * N

        # Pre-compute red-black masks for pressure solver
        sl = (slice(1, -1), slice(1, -1))
        idx = np.arange(N * N).reshape(N, N)
        self._red_mask  = np.zeros(s,  np.float32)
        self._black_mask = np.zeros(s, np.float32)
        self._red_mask[sl]  = ((idx % 2) == 0).astype(np.float32)
        self._black_mask[sl] = ((idx % 2) == 1).astype(np.float32)

        # Splat stencils at multiple radii (for variable brush size)
        self._stencils = {}
        for radius in range(3, 25):
            di = np.arange(-radius, radius + 1, dtype=np.int32)
            dj = np.arange(-radius, radius + 1, dtype=np.int32)
            DI, DJ = np.meshgrid(di, dj, indexing="ij")
            dist = np.sqrt(DI.astype(np.float32) ** 2 + DJ.astype(np.float32) ** 2)
            f = np.maximum(0.0, 1.0 - dist / (radius + 0.5))
            m = f > 0
            self._stencils[radius] = (DI[m], DJ[m], f[m].astype(np.float32))

    def _bnd(self, b, x):
        N = self.N
        if x.ndim == 3:
            x[0, 1:-1] = x[1, 1:-1]; x[-1, 1:-1] = x[-2, 1:-1]
            x[1:-1, 0] = x[1:-1, 1]; x[1:-1, -1] = x[1:-1, -2]
        else:
            x[0, 1:-1]  = -x[1, 1:-1]  if b == 1 else x[1, 1:-1]
            x[-1, 1:-1] = -x[-2, 1:-1] if b == 1 else x[-2, 1:-1]
            x[1:-1, 0]  = -x[1:-1, 1]  if b == 2 else x[1:-1, 1]
            x[1:-1, -1] = -x[1:-1, -2] if b == 2 else x[1:-1, -2]
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

    # ------------------------------------------------------------------
    #  Red-black Jacobi pressure solver  (fully vectorised)
    # ------------------------------------------------------------------
    def _pressure_jacobi_rb(self):
        N = self.N;  h = 1.0 / N
        sl = (slice(1, -1), slice(1, -1))
        div = self._div;  p = self._p;  pn = self._p_new
        div[sl] = -0.5 * h * (
            self.u[2:, 1:-1] - self.u[:-2, 1:-1] +
            self.v[1:-1, 2:] - self.v[1:-1, :-2])
        self._bnd(0, div)
        p[:] = 0
        for _ in range(self.iters):
            # neighbour sum
            ns = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
            pn[sl] = (div[sl] + ns) * 0.25
            # Apply red-black update for faster convergence
            p[sl] = np.where(self._red_mask[sl] != 0, pn[sl], p[sl])
            ns2 = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
            pn[sl] = (div[sl] + ns2) * 0.25
            p[sl] = np.where(self._black_mask[sl] != 0, pn[sl], p[sl])
            self._bnd(0, p)
        self.u[sl] -= 0.5 * N * (p[2:, 1:-1] - p[:-2, 1:-1])
        self.v[sl] -= 0.5 * N * (p[1:-1, 2:] - p[1:-1, :-2])
        self._bnd(1, self.u);  self._bnd(2, self.v)

    # ------------------------------------------------------------------
    def _diffuse(self, b, x, x0, diff, tmp):
        a = self.dt * diff * self.N * self.N
        if a < 1e-12:
            np.copyto(x, x0); return
        c = 1.0 + 4.0 * a;  ci = 1.0 / c;  sl = (slice(1, -1), slice(1, -1))
        for _ in range(self.iters):
            ns = x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]
            tmp[sl] = (x0[sl] + a * ns) * ci
            x[sl] = tmp[sl]
            self._bnd(b, x)

    def _advect(self, b, d, d0, u, v):
        N = self.N;  lo, hi = 0.5, N + 0.5
        x = np.clip(self._adv_ii - self._adv_dtN * u[1:-1, 1:-1], lo, hi)
        y = np.clip(self._adv_jj - self._adv_dtN * v[1:-1, 1:-1], lo, hi)
        i0 = np.floor(x).astype(np.int32);  j0 = np.floor(y).astype(np.int32)
        i1 = i0 + 1;  j1 = j0 + 1
        s1 = x - i0;  s0 = 1.0 - s1;  t1 = y - j0;  t0 = 1.0 - t1
        i0 = np.clip(i0, 0, N);  j0 = np.clip(j0, 0, N)
        i1 = np.clip(i1, 0, N + 1);  j1 = np.clip(j1, 0, N + 1)
        if d.ndim == 2:
            d[1:-1, 1:-1] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1])
                            + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
            self._bnd(b, d)
        else:
            se = s0[..., None]; te = t0[..., None]; s1e = s1[..., None]; t1e = t1[..., None]
            d[1:-1, 1:-1] = (se  * (te  * d0[i0, j0] + t1e * d0[i0, j1])
                            + s1e * (te  * d0[i1, j0] + t1e * d0[i1, j1]))
            self._bnd(0, d)

    def _vorticity_confinement(self, strength=VORTICITY_STR):
        N = self.N;  h = 1.0 / N;  inv2 = 0.5 / h
        sl = (slice(1, -1), slice(1, -1))
        w = self._tmp2c;  gx = self._tmp2a;  gy = self._tmp2b;  length = self._div
        w[sl] = ((self.v[2:, 1:-1] - self.v[:-2, 1:-1])
                - (self.u[1:-1, 2:] - self.u[1:-1, :-2])) * inv2
        ac = np.abs(w)
        gx[sl] = (ac[2:, 1:-1] - ac[:-2, 1:-1]) * inv2
        gy[sl] = (ac[1:-1, 2:] - ac[1:-1, :-2]) * inv2
        np.multiply(gx, gx, out=length); np.multiply(gy, gy, out=gy)
        np.add(length, gy, out=length); np.sqrt(length, out=length)
        length[sl] += 1e-5;  gx[sl] /= length[sl];  gy[sl] /= length[sl]
        sh = strength * h
        self.u[sl] += sh * gy[sl] * w[sl]
        self.v[sl] -= sh * gx[sl] * w[sl]

    # ------------------------------------------------------------------
    #  Buoyancy  (red rises, blue sinks)
    # ------------------------------------------------------------------
    def _apply_buoyancy(self, buoy=BUOYANCY_STRENGTH, grav=GRAVITY_STRENGTH):
        # Temperature proxy: red=hot (rises), blue=cold (sinks)
        temp = self.d[1:-1, 1:-1, 0] - self.d[1:-1, 1:-1, 2]
        self.v[1:-1, 1:-1] -= buoy * temp

    # ------------------------------------------------------------------
    #  Injection (variable radius)
    # ------------------------------------------------------------------
    def _get_stencil(self, radius):
        r = max(3, min(24, int(radius)))
        if r not in self._stencils:
            r = max(3, min(24, r))
        return self._stencils[r]

    def add_density(self, cx, cy, amount, color, radius=5):
        N = self.N
        di, dj, fv = self._get_stencil(radius)
        ni = cx + di;  nj = cy + dj
        ok = (ni >= 1) & (ni <= N) & (nj >= 1) & (nj <= N)
        ni, nj, fv = ni[ok], nj[ok], fv[ok]
        c = np.array(color, dtype=np.float32)
        self.d0[ni, nj] += (np.float32(amount) * fv)[:, None] * c[None, :]

    def add_velocity(self, cx, cy, vx, vy, radius=5):
        N = self.N
        di, dj, fv = self._get_stencil(radius)
        ni = cx + di;  nj = cy + dj
        ok = (ni >= 1) & (ni <= N) & (nj >= 1) & (nj <= N)
        ni, nj, fv = ni[ok], nj[ok], fv[ok]
        self.u0[ni, nj] += np.float32(vx) * fv
        self.v0[ni, nj] += np.float32(vy) * fv

    def attract(self, cx, cy, strength=ATTRACT_FORCE, radius=15):
        """Pull surrounding fluid toward (cx, cy)."""
        N = self.N
        di, dj, fv = self._get_stencil(radius)
        ni = cx + di;  nj = cy + dj
        ok = (ni >= 1) & (ni <= N) & (nj >= 1) & (nj <= N)
        ni, nj, fv = ni[ok], nj[ok], fv[ok]
        dx = (cx - ni).astype(np.float32)
        dy = (cy - nj).astype(np.float32)
        dist = np.sqrt(dx * dx + dy * dy + 1.0)
        self.u0[ni, nj] += strength * dx / dist * fv
        self.v0[ni, nj] += strength * dy / dist * fv

    def repel(self, cx, cy, strength=REPEL_FORCE, radius=15):
        """Push surrounding fluid away from (cx, cy)."""
        N = self.N
        di, dj, fv = self._get_stencil(radius)
        ni = cx + di;  nj = cy + dj
        ok = (ni >= 1) & (ni <= N) & (nj >= 1) & (nj <= N)
        ni, nj, fv = ni[ok], nj[ok], fv[ok]
        dx = (ni - cx).astype(np.float32)
        dy = (nj - cy).astype(np.float32)
        dist = np.sqrt(dx * dx + dy * dy + 1.0)
        self.u0[ni, nj] += strength * dx / dist * fv
        self.v0[ni, nj] += strength * dy / dist * fv

    # ------------------------------------------------------------------
    #  Full step
    # ------------------------------------------------------------------
    def step(self, buoyancy_on=False):
        self.u += self.dt * self.u0;  self.v += self.dt * self.v0
        np.clip(self.u, -VELOCITY_CAP, VELOCITY_CAP, out=self.u)
        np.clip(self.v, -VELOCITY_CAP, VELOCITY_CAP, out=self.v)

        if buoyancy_on:
            self._apply_buoyancy()

        # Velocity solve
        self.u, self.u0 = self.u0, self.u
        self._diffuse(1, self.u, self.u0, self.visc, self._tmp2a)
        self.v, self.v0 = self.v0, self.v
        self._diffuse(2, self.v, self.v0, self.visc, self._tmp2a)
        self._pressure_jacobi_rb()

        self.u0[:] = self.u;  self.v0[:] = self.v
        self._advect(1, self.u, self.u0, self.u0, self.v0)
        self._advect(2, self.v, self.v0, self.u0, self.v0)
        self._pressure_jacobi_rb()
        self._vorticity_confinement()
        self.u0[:] = 0;  self.v0[:] = 0

        # Density solve
        self.d += self.dt * self.d0
        self.d, self.d0 = self.d0, self.d
        self._diffuse(0, self.d, self.d0, self.diff, self._tmp3)
        self.d0[:] = self.d
        self._advect(0, self.d, self.d0, self.u, self.v)
        self.d0[:] = 0
        self.d *= DENSITY_DECAY

        np.nan_to_num(self.d, copy=False)
        np.nan_to_num(self.u, copy=False)
        np.nan_to_num(self.v, copy=False)

    def kinetic_energy(self):
        sl = (slice(1, -1), slice(1, -1))
        return float(np.sum(self.u[sl] ** 2 + self.v[sl] ** 2))

    def reset(self):
        for a in (self.u, self.v, self.d, self.u0, self.v0, self.d0):
            a[:] = 0


# ======================================================================
#  RENDERER
# ======================================================================

class FluidRenderer:
    def __init__(self, N):
        self.N = N
        self._rows = np.arange(N, dtype=np.int32)[:, None]
        self._cols = np.arange(N, dtype=np.int32)[None, :]
        self._stk  = np.zeros((4, N, N), np.float32)
        self._rgb  = np.zeros((N, N, 3), np.float32)
        self._u8   = np.zeros((N, N, 3), np.uint8)
        self._val  = np.zeros((N, N), np.float32)
        self._r = np.zeros((N, N), np.float32)
        self._g = np.zeros((N, N), np.float32)
        self._b = np.zeros((N, N), np.float32)
        y, x = np.mgrid[0:N, 0:N]
        dist = np.sqrt((x - N / 2.0) ** 2 + (y - N / 2.0) ** 2) / (N / 2.0)
        self._vig = np.clip(1.0 - dist * 0.7, 0.2, 1.0).astype(np.float32)[:, :, None]
        self._cx = (x - N / 2.0).astype(np.float32) / (N / 2.0)
        self._cy = (y - N / 2.0).astype(np.float32) / (N / 2.0)
        self._angle = np.arctan2(self._cy, self._cx).astype(np.float32)
        self._radius = np.sqrt(self._cx ** 2 + self._cy ** 2).astype(np.float32)
        self._thermal_lut = self._build_thermal_lut()
        self._noise = self._make_noise(N)
        # Bloom buffers (half-res for speed)
        self._bloom_w = max(1, N // 4)
        self._bloom_h = max(1, N // 4)
        self._bloom_a = np.zeros((self._bloom_h, self._bloom_w, 3), np.float32)
        self._bloom_b = np.zeros_like(self._bloom_a)

    @staticmethod
    def _build_thermal_lut():
        lut = np.zeros((256, 3), np.float32)
        stops = [
            (0,   (0.0,  0.0,  0.05)), (40,  (0.0,  0.0,  0.6)),
            (90,  (0.0,  0.6,  0.9)),   (140, (0.1,  0.9,  0.2)),
            (190, (0.95, 0.9,  0.1)),  (225, (1.0,  0.3,  0.05)),
            (255, (1.0,  1.0,  1.0)),
        ]
        for i in range(len(stops) - 1):
            i0, c0 = stops[i];  i1, c1 = stops[i + 1]
            for j in range(i0, i1 + 1):
                f = (j - i0) / max(1, i1 - i0)
                lut[j] = [c0[k] + (c1[k] - c0[k]) * f for k in range(3)]
        return lut

    @staticmethod
    def _make_noise(N, scale=6):
        rng = np.random.RandomState(42)
        noise = np.zeros((N, N), np.float32)
        for octave in range(4):
            s = scale * (2 ** octave)
            small = rng.rand(s, s).astype(np.float32)
            yi = np.linspace(0, s - 1, N).astype(np.float32)
            xi = np.linspace(0, s - 1, N).astype(np.float32)
            y0 = np.floor(yi).astype(int);  x0 = np.floor(xi).astype(int)
            y1 = np.minimum(y0 + 1, s - 1); x1 = np.minimum(x0 + 1, s - 1)
            fy = (yi - y0)[:, None];  fx = (xi - x0)[None, :]
            noise += (small[y0][:, x0] * (1 - fy) * (1 - fx)
                    + small[y1][:, x0] * fy * (1 - fx)
                    + small[y0][:, x1] * (1 - fy) * fx
                    + small[y1][:, x1] * fy * fx) / (2 ** octave)
        mn, mx = noise.min(), noise.max()
        return ((noise - mn) / (mx - mn + 1e-8)).astype(np.float32)

    def _apply_bloom(self, out, strength=0.35):
        """Cheap bloom: downsample -> box blur twice -> upsample + add."""
        if strength < 0.01:
            return
        bh, bw = self._bloom_h, self._bloom_w
        # Downsample
        N = self.N
        step_y = max(1, N // bh)
        step_x = max(1, N // bw)
        small = out[::step_y, ::step_x][:bh, :bw]
        ba = self._bloom_a;  bb = self._bloom_b
        ba[:] = small
        # Two-pass box blur
        for _ in range(2):
            bb[1:-1, :] = (ba[:-2, :] + ba[1:-1, :] + ba[2:, :]) / 3.0
            ba[:, 1:-1] = (bb[:, :-2] + bb[:, 1:-1] + bb[:, 2:]) / 3.0
        # Upsample and add
        bloom_full = np.repeat(np.repeat(ba, step_y, axis=0), step_x, axis=1)[:N, :N]
        if bloom_full.shape != (N, N, 3):
            bloom_full = bloom_full[:N, :N, :]
        out += bloom_full * strength

    def process(self, density, u, v, t, mode, bloom_on=True, bloom_str=0.35):
        N = self.N;  out = self._rgb
        np.nan_to_num(density, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
        np.nan_to_num(u, nan=0.0, copy=False);  np.nan_to_num(v, nan=0.0, copy=False)
        di = density[1:-1, 1:-1];  ui = u[1:-1, 1:-1];  vi = v[1:-1, 1:-1]
        speed = np.sqrt(ui * ui + vi * vi)
        dsum = np.sum(di, axis=-1, out=self._val)

        if mode == 0:  # Vibrant Oil
            ar = (density[:-2, 1:-1] + density[2:, 1:-1]
                + density[1:-1, :-2] + density[1:-1, 2:]) * 0.25
            dr = di[:,:,0] + (di[:,:,0] - ar[:,:,0]) * 0.3
            dg = di[:,:,1] + (di[:,:,1] - ar[:,:,1]) * 0.3
            db = di[:,:,2] + (di[:,:,2] - ar[:,:,2]) * 0.3
            avg = (dr + dg + db) / 3.0
            dr += (dr - avg) * 1.5;  dg += (dg - avg) * 1.5;  db += (db - avg) * 1.5
            dr /= (1 + dr);  dg /= (1 + dg);  db /= (1 + db)
            sn = np.minimum(1.0, speed * 0.4)
            out[:,:,0] = dr + sn*0.15;  out[:,:,1] = dg + sn*0.15;  out[:,:,2] = db + sn*0.2
            np.divide(out, 1+out, out=out);  out *= self._vig
            shift = int(3 + 2 * np.sin(t * 0.5))
            if shift > 0: out[:-shift,:,0] = out[shift:,:,0]; out[shift:,:,2] = out[:-shift,:,2]
            if bloom_on: self._apply_bloom(out, bloom_str)
            np.clip(out, 0, 1, out=out);  np.sqrt(out, out=out);  np.multiply(out, 255, out=out)

        elif mode == 1:  # Neon Glow
            angle = np.arctan2(vi, ui)
            hue = ((angle / 6.2832) + 0.5 + t*0.05 + di[:,:,0]*0.05) % 1.0
            sat = np.clip(speed*4+0.5, 0.6, 1.0);  val = dsum + speed*1.5
            np.clip(val, 0, 1.5, out=val)
            _hsv_to_rgb_vec(hue, sat, val, self._stk, self._rgb, self._rows, self._cols)
            self._rgb[val > 1.0] += 0.2
            np.divide(out, 1+out, out=out);  out *= self._vig
            if bloom_on: self._apply_bloom(out, bloom_str * 0.8)
            np.clip(out, 0, 1, out=out);  np.sqrt(out, out=out);  np.multiply(out, 255, out=out)

        elif mode == 2:  # Velocity Field
            out[:,:,0] = np.clip(ui*5+0.5, 0, 1);  out[:,:,1] = np.clip(vi*5+0.5, 0, 1)
            out[:,:,2] = np.clip((ui+vi)*2.5+0.5, 0, 1);  out += 0.1
            out *= self._vig;  np.clip(out, 0, 1, out=out)
            np.sqrt(out, out=out);  np.multiply(out, 255, out=out)

        elif mode == 3:  # Thermal
            intensity = np.clip(dsum*2.5 + speed*0.8, 0, 1)
            idx = np.clip((intensity*255).astype(np.int32), 0, 255)
            out[:,:,0] = self._thermal_lut[idx,0]
            out[:,:,1] = self._thermal_lut[idx,1]
            out[:,:,2] = self._thermal_lut[idx,2]
            bloom = np.maximum(0, intensity - 0.6) * 2.5
            out[:,:,0] += bloom*0.3;  out[:,:,1] += bloom*0.2
            out *= self._vig
            if bloom_on: self._apply_bloom(out, bloom_str)
            np.clip(out, 0, 1, out=out);  np.power(out, 0.85, out=out);  np.multiply(out, 255, out=out)

        elif mode == 4:  # Plasma
            gx = np.zeros((N,N), np.float32);  gy = np.zeros_like(gx)
            gx[1:-1,:] = (di[2:,:,0] - di[:-2,:,0]) * 0.5
            gy[:,1:-1] = (di[:,2:,0] - di[:,:-2,0]) * 0.5
            grad = np.sqrt(gx*gx + gy*gy + 1e-10)
            glow = np.clip(grad*8 + speed*2, 0, 2)
            angle = np.arctan2(vi, ui)
            hue = ((angle/6.2832)+0.5+t*0.08+di[:,:,1]*0.3-di[:,:,0]*0.3) % 1.0
            sat = np.clip(0.7+dsum*0.5, 0.5, 1.0)
            val = np.power(np.clip(glow, 0, 1), 1.5)
            _hsv_to_rgb_vec(hue, sat, val, self._stk, self._rgb, self._rows, self._cols)
            core = np.clip(dsum-0.5, 0, 1) * 0.4
            out += core[:,:,None];  out *= self._vig
            if bloom_on: self._apply_bloom(out, bloom_str * 1.2)
            np.clip(out, 0, 1, out=out);  np.power(out, 0.8, out=out);  np.multiply(out, 255, out=out)

        elif mode == 5:  # Ink Drop
            r = 0.06+di[:,:,0]*0.6+di[:,:,1]*0.15
            g = 0.08+di[:,:,1]*0.6+di[:,:,2]*0.15
            b = 0.12+di[:,:,2]*0.6+di[:,:,0]*0.15
            gx = np.zeros((N,N), np.float32); gy = np.zeros_like(gx)
            gx[1:-1,:] = (di[2:,:,0]-di[:-2,:,0])*0.5
            gy[:,1:-1] = (di[:,2:,0]-di[:,:-2,0])*0.5
            gm = np.sqrt(gx*gx+gy*gy)
            eb = np.clip(gm*2, 0, 0.15)
            r += eb*0.3;  g += eb*0.35;  b += eb*0.5
            ripple = np.clip(speed*0.3, 0, 0.08)
            r += ripple*0.5;  g += ripple*0.7;  b += ripple*1.0
            out[:,:,0]=r; out[:,:,1]=g; out[:,:,2]=b
            out[:,:,0] /= (1+out[:,:,0]); out[:,:,1] /= (1+out[:,:,1]); out[:,:,2] /= (1+out[:,:,2])
            out *= self._vig;  out -= (1-self._vig)*0.03
            np.clip(out, 0, 1, out=out);  np.power(out, 0.9, out=out);  np.multiply(out, 255, out=out)

        elif mode == 6:  # Prismatic
            grx = np.zeros((N,N), np.float32); gry = np.zeros_like(grx)
            grx[1:-1,:] = (di[2:,:,0]-di[:-2,:,0])*0.5
            gry[:,1:-1] = (di[:,2:,0]-di[:,:-2,0])*0.5
            ga = np.arctan2(gry, grx)
            hue = ((ga/6.2832)+0.5+self._radius*0.3+t*0.06+dsum*0.1) % 1.0
            sat = np.clip(dsum*1.2+0.1, 0, 1)
            sparkle = np.abs(np.sin(ga*5+t*3))
            val = np.clip(dsum+speed*0.5+sparkle*dsum*0.3, 0, 1.2)
            _hsv_to_rgb_vec(hue, sat, val, self._stk, self._rgb, self._rows, self._cols)
            ds = int(2 + np.sin(t*0.3)*2)
            if ds > 0: out[:-ds,:,0]=out[ds:,:,0]*0.85; out[ds:,:,2]=out[:-ds,:,2]*0.85
            out *= self._vig
            if bloom_on: self._apply_bloom(out, bloom_str * 0.6)
            np.clip(out, 0, 1, out=out);  np.power(out, 0.85, out=out);  np.multiply(out, 255, out=out)

        elif mode == 7:  # Smoke
            sv = np.clip(dsum*1.2, 0, 1)
            turb = self._noise*0.4+0.6
            ph = t*0.15
            drift = np.sin(self._cx*3+ph)*np.cos(self._cy*2.5+ph*0.7)*0.2+0.8
            sv *= turb * drift
            sv = np.clip(sv + speed*0.045, 0, 1)
            warm = np.clip(ui*0.5+0.5, 0, 1)
            r = sv*(0.65+warm*0.35);  g = sv*(0.62+warm*0.15);  b = sv*(0.70-warm*0.15)
            lap = np.zeros((N,N), np.float32)
            lap[1:-1,1:-1] = (-4*dsum[1:-1,1:-1]+dsum[:-2,1:-1]+dsum[2:,1:-1]+dsum[1:-1,:-2]+dsum[1:-1,2:])
            th = np.clip(-lap*3, 0, 0.3)
            r += th*0.2;  g += th*0.18;  b += th*0.25
            out[:,:,0]=np.clip(r+0.02,0,1); out[:,:,1]=np.clip(g+0.02,0,1); out[:,:,2]=np.clip(b+0.03,0,1)
            out *= self._vig*(0.4+0.6)
            np.clip(out, 0, 1, out=out);  np.power(out, 0.92, out=out);  np.multiply(out, 255, out=out)
        else:
            out[:] = 0.1;  np.multiply(out, 255, out=out)

        self._u8[:] = out
        return self._u8


# ======================================================================
#  AUDIO  (optional simple sine tone mapped to fluid energy)
# ======================================================================

class FluidAudio:
    """Minimal audio feedback: sine tone pitch/volume tracks fluid energy."""
    def __init__(self):
        self.enabled = False
        self._freq = 220.0
        self._vol  = 0.0
        self._phase = 0.0
        self._sample_rate = 22050
        self._buf_size = 512
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=self._sample_rate, size=-16,
                               channels=1, buffer=self._buf_size)
            self._available = True
            self._snd = pygame.sndarray.make_sound(
                np.zeros((self._buf_size,), dtype=np.int16))
            self._snd.play(loops=-1)
        except Exception:
            self._available = False

    def update(self, energy, dt):
        if not self.enabled or not self._available:
            if self._available:
                self._write_silence()
            return
        # Map energy to frequency (100-800 Hz) and volume (0-0.3)
        target_freq = 100.0 + min(energy * 0.5, 700.0)
        target_vol  = min(0.25, energy * 0.002)
        self._freq += (target_freq - self._freq) * 0.1
        self._vol  += (target_vol  - self._vol)  * 0.15
        # Generate buffer
        n = self._buf_size
        t = np.arange(n, dtype=np.float32) / self._sample_rate
        phase_inc = 2.0 * np.pi * self._freq
        samples = np.sin(self._phase + t * phase_inc) * self._vol
        # Add subtle harmonics
        samples += np.sin(self._phase + t * phase_inc * 2.0) * self._vol * 0.3
        samples += np.sin(self._phase + t * phase_inc * 0.5) * self._vol * 0.2
        self._phase += t[-1] * phase_inc
        samples = np.clip(samples * 32767, -32767, 32767).astype(np.int16)
        try:
            arr = pygame.sndarray.array(samples)
            self._snd = pygame.sndarray.make_sound(arr)
            self._snd.play()
        except Exception:
            pass

    def _write_silence(self):
        try:
            samples = np.zeros((self._buf_size,), dtype=np.int16)
            arr = pygame.sndarray.array(samples)
            self._snd = pygame.sndarray.make_sound(arr)
            self._snd.play()
        except Exception:
            pass

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self._vol = 0.0
        return self.enabled


# ======================================================================
#  UI BUTTON
# ======================================================================

class UIButton:
    def __init__(self, x, y, w, h, text, font,
                 bg=(50,50,50), hover_bg=(70,70,70),
                 fg=(220,220,220), active_bg=(60,120,180)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text;  self.font = font
        self.bg = bg;  self.hover_bg = hover_bg;  self.fg = fg
        self.active_bg = active_bg;  self.active = False;  self.hovered = False

    def draw(self, surface):
        c = self.active_bg if self.active else (self.hover_bg if self.hovered else self.bg)
        pygame.draw.rect(surface, c, self.rect, border_radius=4)
        pygame.draw.rect(surface, (90,90,90), self.rect, 1, border_radius=4)
        txt = self.font.render(self.text, True, self.fg)
        surface.blit(txt, (self.rect.x + (self.rect.w - txt.get_width())//2,
                           self.rect.y + (self.rect.h - txt.get_height())//2))

    def update_hover(self, mx, my):
        self.hovered = self.rect.collidepoint(mx, my)

    def clicked(self, mx, my):
        return self.rect.collidepoint(mx, my)


# ======================================================================
#  MAIN APPLICATION
# ======================================================================

class LiquidVizApp:
    def __init__(self):
        pygame.init()
        self.N = SIM_GRID_SIZE
        self.win_size = WINDOW_SIZE
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode(
            (self.win_size, self.win_size + 55))
        pygame.display.set_caption("LiquidViz - Pygame Fluid Simulation")
        self.clock = pygame.time.Clock()

        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.font_md = pygame.font.SysFont("consolas", 15, bold=True)

        self.solver   = FluidSolver(self.N)
        self.renderer = FluidRenderer(self.N)
        self.audio    = FluidAudio()

        # State
        self.preset_names  = list(PRESETS.keys())
        self.preset_idx    = 0
        self.render_mode   = 0
        self.is_paused     = False
        self.sim_time      = 0.0
        self.speed_mult    = 1.0
        self.seeds         = [random.random()*100 for _ in range(20)]
        self.frame_times   = []
        self.fps_display   = "-- FPS"
        self.bloom_on      = True
        self.bloom_str     = 0.35
        self.buoyancy_on   = False
        self.brush_radius  = 5   # grid cells
        self.screenshot_msg = ""
        self.screenshot_timer = 0.0

        # Mouse
        self._mouse_down  = False
        self._mmouse_down = False
        self._shift_held  = False
        self._last_gx = 0;  self._last_gy = 0

        # UI
        self._build_toolbar()

        # Warmup
        self.status_str = "Warming up..."
        self._warmup(WARMUP_STEPS, WARMUP_SPEED)
        self.status_str = self.preset_names[self.preset_idx]

    def _build_toolbar(self):
        bar_y = self.win_size + 3
        bx = 8;  by = bar_y + 3;  bh = 28;  gap = 5
        self.btn_preset = UIButton(bx, by, 135, bh,
            "Preset: " + self.preset_names[0], self.font_sm, active_bg=(40,100,60))
        bx += 135 + gap
        self.btn_mode = UIButton(bx, by, 120, bh,
            "Vis:" + RENDER_MODES[0], self.font_sm)
        bx += 120 + gap
        self.btn_pause = UIButton(bx, by, 65, bh, "Pause", self.font_sm)
        bx += 65 + gap
        self.btn_reset = UIButton(bx, by, 60, bh, "Reset", self.font_sm)
        bx += 60 + gap
        self.btn_speed = UIButton(bx, by, 78, bh, "Spd 1.0x", self.font_sm)
        bx += 78 + gap
        self.btn_bloom = UIButton(bx, by, 60, bh, "Bloom", self.font_sm, active_bg=(100,60,40))
        self.btn_bloom.active = True
        bx += 60 + gap
        self.btn_grav = UIButton(bx, by, 55, bh, "Grav", self.font_sm)
        bx += 55 + gap
        self.btn_audio = UIButton(bx, by, 50, bh, "Audio", self.font_sm)
        self.all_buttons = [
            self.btn_preset, self.btn_mode, self.btn_pause, self.btn_reset,
            self.btn_speed, self.btn_bloom, self.btn_grav, self.btn_audio,
        ]

    # ------------------------------------------------------------------
    #  Preset injection (curl-noise driven)
    # ------------------------------------------------------------------
    def _inject_autonomous(self):
        name = self.preset_names[self.preset_idx]
        p = PRESETS[name];  N = self.N;  cc = p["colors"];  num = p["num_sources"]
        for i in range(num):
            seed = self.seeds[i]
            t = self.sim_time * p["speed"] + seed
            fx = 2.0 + (i % 3) * 0.7;  fy = 3.0 + (i % 4) * 0.6
            ph = seed * 6.2832;  rad = p["radius"]
            cx = 0.5 + rad * np.sin(t * fx + ph)
            cy = 0.5 + rad * np.cos(t * fy)
            # Add curl noise perturbation for more organic motion
            curl_x, curl_y = _curl_noise_2d(
                np.float32(cx * 4.0), np.float32(cy * 4.0),
                seed=seed, freq=2.0 + i * 0.3)
            gx = int(cx * N);  gy = int(cy * N)
            if 1 <= gx <= N and 1 <= gy <= N:
                vx = (np.cos(t*fx+ph)*fx*rad*p["force"]
                      + curl_x * 3.0 + np.sin(t*50)*1.5)
                vy = (-np.sin(t*fy)*fy*rad*p["force"]
                      + curl_y * 3.0 + np.sin(t*50)*1.5)
                if cc == "rainbow":
                    color = colorsys.hsv_to_rgb(
                        (self.sim_time*0.1 + i*0.1) % 1.0, 1.0, 1.0)
                else:
                    color = cc[int(self.sim_time*0.5+i) % len(cc)]
                self.solver.add_density(gx, gy, p["density"], color, p["brush_size"])
                self.solver.add_velocity(gx, gy, vx, vy, p["brush_size"])

    # ------------------------------------------------------------------
    #  Mouse
    # ------------------------------------------------------------------
    def _screen_to_grid(self, mx, my):
        N = self.N;  ws = self.win_size
        gx = int(mx / ws * N) + 1;  gy = int(my / ws * N) + 1
        return max(1, min(N, gx)), max(1, min(N, gy))

    def _inject_mouse(self, gx, gy, pgx, pgy, color=None):
        if color is None: color = MOUSE_COLOR
        vx = (gx - pgx) * MOUSE_FORCE;  vy = (gy - pgy) * MOUSE_FORCE
        if (vx*vx + vy*vy)**0.5 < 1.0:
            a = ((gx+gy)*0.1) % 6.2832
            vx = math.cos(a)*MOUSE_FORCE*2;  vy = math.sin(a)*MOUSE_FORCE*2
        self.solver.add_density(gx, gy, MOUSE_DENSITY, color, self.brush_radius)
        self.solver.add_velocity(gx, gy, vx, vy, self.brush_radius)

    # ------------------------------------------------------------------
    def _warmup(self, steps, speed_mult):
        for _ in range(steps):
            self.sim_time += (1.0/FPS_CAP) * speed_mult
            self._inject_autonomous()
            self.solver.step(self.buoyancy_on)

    # ------------------------------------------------------------------
    #  Screenshot
    # ------------------------------------------------------------------
    def _save_screenshot(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"liquidviz_{ts}.png")
        pygame.image.save(self.screen, path)
        self.screenshot_msg = f"Saved: {os.path.basename(path)}"
        self.screenshot_timer = 2.0

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            dt_real = self.clock.get_time() / 1000.0
            mx, my = pygame.mouse.get_pos()
            keys = pygame.key.get_mods()
            self._shift_held = bool(keys & pygame.KMOD_SHIFT)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k == pygame.K_ESCAPE:   running = False
                    elif k == pygame.K_SPACE:
                        self.is_paused = not self.is_paused
                        self.btn_pause.text = "Resume" if self.is_paused else "Pause"
                    elif k == pygame.K_r:       self._do_reset()
                    elif k == pygame.K_c:       self.solver.d[:] = 0
                    elif k == pygame.K_TAB:     self._cycle_preset()
                    elif k == pygame.K_b:
                        self.bloom_on = not self.bloom_on
                        self.btn_bloom.active = self.bloom_on
                    elif k == pygame.K_g:
                        self.buoyancy_on = not self.buoyancy_on
                        self.btn_grav.active = self.buoyancy_on
                    elif k == pygame.K_m:
                        on = self.audio.toggle()
                        self.btn_audio.active = on
                    elif k == pygame.K_F11:
                        self._toggle_fullscreen()
                    elif k == pygame.K_F12:
                        self._save_screenshot()
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.speed_mult = min(4.0, self.speed_mult + 0.5)
                        self.btn_speed.text = f"Spd {self.speed_mult:.1f}x"
                    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.speed_mult = max(0.5, self.speed_mult - 0.5)
                        self.btn_speed.text = f"Spd {self.speed_mult:.1f}x"
                    elif pygame.K_1 <= k <= pygame.K_9:
                        idx = k - pygame.K_1
                        if idx < len(RENDER_MODES):
                            self.render_mode = idx
                            self.btn_mode.text = "Vis:" + RENDER_MODES[idx]

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked_ui = False
                        if   self.btn_preset.clicked(mx,my): self._cycle_preset(); clicked_ui=True
                        elif self.btn_mode.clicked(mx,my):
                            self.render_mode = (self.render_mode+1)%len(RENDER_MODES)
                            self.btn_mode.text = "Vis:" + RENDER_MODES[self.render_mode]
                            clicked_ui=True
                        elif self.btn_pause.clicked(mx,my):
                            self.is_paused = not self.is_paused
                            self.btn_pause.text = "Resume" if self.is_paused else "Pause"
                            clicked_ui=True
                        elif self.btn_reset.clicked(mx,my): self._do_reset(); clicked_ui=True
                        elif self.btn_speed.clicked(mx,my):
                            speeds=[0.5,1.0,1.5,2.0,3.0,4.0]
                            idx=speeds.index(self.speed_mult) if self.speed_mult in speeds else 1
                            self.speed_mult=speeds[(idx+1)%len(speeds)]
                            self.btn_speed.text=f"Spd {self.speed_mult:.1f}x"
                            clicked_ui=True
                        elif self.btn_bloom.clicked(mx,my):
                            self.bloom_on=not self.bloom_on; self.btn_bloom.active=self.bloom_on; clicked_ui=True
                        elif self.btn_grav.clicked(mx,my):
                            self.buoyancy_on=not self.buoyancy_on; self.btn_grav.active=self.buoyancy_on; clicked_ui=True
                        elif self.btn_audio.clicked(mx,my):
                            on=self.audio.toggle(); self.btn_audio.active=on; clicked_ui=True
                        if not clicked_ui and my < self.win_size:
                            self._mouse_down = True
                            gx, gy = self._screen_to_grid(mx, my)
                            self._last_gx, self._last_gy = gx, gy
                            self._inject_mouse(gx, gy, gx, gy)
                    elif event.button == 2 and my < self.win_size:  # Middle
                        self._mmouse_down = True
                        gx, gy = self._screen_to_grid(mx, my)
                        self._last_gx, self._last_gy = gx, gy
                    elif event.button == 3 and my < self.win_size:
                        gx, gy = self._screen_to_grid(mx, my)
                        rc = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
                        self._inject_mouse(gx, gy, gx, gy, color=rc)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  self._mouse_down = False
                    elif event.button == 2: self._mmouse_down = False

                elif event.type == pygame.MOUSEMOTION:
                    if my < self.win_size:
                        gx, gy = self._screen_to_grid(mx, my)
                        if self._mouse_down and not self.is_paused:
                            if self._shift_held:
                                self.solver.repel(gx, gy, REPEL_FORCE, self.brush_radius * 3)
                            else:
                                self._inject_mouse(gx, gy, self._last_gx, self._last_gy)
                        if self._mmouse_down and not self.is_paused:
                            self.solver.attract(gx, gy, ATTRACT_FORCE, self.brush_radius * 3)
                        self._last_gx, self._last_gy = gx, gy

                elif event.type == pygame.MOUSEWHEEL:
                    self.brush_radius = max(3, min(24, self.brush_radius + event.y))

                elif event.type == pygame.VIDEORESIZE and self.is_fullscreen:
                    self.win_size = min(event.w, event.h - 55)

            # ---- Simulate ----
            if not self.is_paused:
                steps = max(1, int(self.speed_mult))
                for _ in range(steps):
                    self.sim_time += 1.0 / FPS_CAP
                    self._inject_autonomous()
                    self.solver.step(self.buoyancy_on)

            # ---- Audio ----
            energy = self.solver.kinetic_energy()
            self.audio.update(energy, dt_real)

            # ---- Render ----
            img_u8 = self.renderer.process(
                self.solver.d, self.solver.u, self.solver.v,
                self.sim_time, self.render_mode,
                self.bloom_on, self.bloom_str)
            surf = pygame.surfarray.make_surface(img_u8)
            scaled = pygame.transform.smoothscale(surf, (self.win_size, self.win_size))
            self.screen.blit(scaled, (0, 0))

            # ---- Brush cursor ----
            if my < self.win_size:
                bpx = int(self.brush_radius / self.N * self.win_size * 2)
                cursor_color = (255,100,100,120) if self._shift_held else (
                    (100,200,255,120) if self._mmouse_down else (200,200,200,80))
                cursor_surf = pygame.Surface((bpx*2, bpx*2), pygame.SRCALPHA)
                pygame.draw.circle(cursor_surf, cursor_color, (bpx, bpx), bpx, 1)
                self.screen.blit(cursor_surf, (mx - bpx, my - bpx))

            # ---- FPS ----
            now = time.perf_counter()
            self.frame_times.append(now)
            if len(self.frame_times) > 90: self.frame_times.pop(0)
            if len(self.frame_times) > 1:
                dt_avg = (self.frame_times[-1]-self.frame_times[0])/(len(self.frame_times)-1)
                self.fps_display = f"{int(1/dt_avg) if dt_avg>0 else 0} FPS"

            # ---- HUD ----
            parts = [self.preset_names[self.preset_idx]]
            if self.is_paused: parts.append("PAUSED")
            if self.speed_mult != 1.0: parts.append(f"{self.speed_mult:.1f}x")
            if self.buoyancy_on: parts.append("BUOY")
            if self.bloom_on: parts.append("BLOOM")
            if self.audio.enabled: parts.append("AUDIO")
            self.status_str = " | ".join(parts)

            self.screen.blit(self.font_sm.render(self.fps_display, True, (255,255,255)), (10, 6))
            self.screen.blit(self.font_sm.render(self.status_str, True, (200,200,200)), (10, 22))
            self.screen.blit(self.font_sm.render(
                f"Brush:{self.brush_radius}", True, (160,160,160)), (10, 38))

            # Screenshot flash
            if self.screenshot_timer > 0:
                self.screenshot_timer -= dt_real
                self.screen.blit(self.font_sm.render(
                    self.screenshot_msg, True, (100,255,100)), (self.win_size - 220, 8))

            # ---- Toolbar ----
            bar_rect = pygame.Rect(0, self.win_size, self.screen.get_width(), 55)
            pygame.draw.rect(self.screen, (30,30,30), bar_rect)
            for btn in self.all_buttons:
                btn.update_hover(mx, my);  btn.draw(self.screen)
            help_text = ("LMB:inject  RMB:color  MMB:attract  Shift+LMB:repel  "
                         "Scroll:brush  F11:fullscreen  F12:screenshot")
            self.screen.blit(self.font_sm.render(help_text, True, (90,90,90)),
                             (8, self.win_size + 38))

            pygame.display.flip()
            self.clock.tick(FPS_CAP)

        pygame.quit()

    # ------------------------------------------------------------------
    def _toggle_fullscreen(self):
        if not self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            self.win_size = min(info.current_w, info.current_h - 55)
            self.is_fullscreen = True
        else:
            self.win_size = WINDOW_SIZE
            self.screen = pygame.display.set_mode((self.win_size, self.win_size + 55))
            self.is_fullscreen = False
        self._build_toolbar()

    def _cycle_preset(self):
        self.preset_idx = (self.preset_idx + 1) % len(self.preset_names)
        self.btn_preset.text = "Preset: " + self.preset_names[self.preset_idx]
        self.solver.reset()
        self.seeds = [random.random()*100 for _ in range(20)]
        self.sim_time = 0.0
        self._warmup(WARMUP_STEPS // 2, 2.0)

    def _do_reset(self):
        self.solver.reset()
        self.seeds = [random.random()*100 for _ in range(20)]
        self.sim_time = 0.0
        self._warmup(WARMUP_STEPS // 2, 2.0)


def main():
    app = LiquidVizApp()
    app.run()

if __name__ == "__main__":
    main()