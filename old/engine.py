# engine.py - GPU fluid simulation engine (numba.cuda backend)
#
# CRITICAL FIX: Original used cupy arrays with numba.cuda kernels —
# fundamentally incompatible.  Now uses numba.cuda device arrays throughout.
#
# CRITICAL FIX: Pressure solver no longer allocates a new buffer every frame.
# Uses pre-allocated double-buffer (self.p, self.p_new) with SOR for
# faster convergence.

import math

from numba import cuda
import numpy as np
from config import (
    SIM_GRID_SIZE, ITERATIONS, DISSIPATION, VELOCITY_DISSIPATION,
    VORTICITY_STRENGTH, SOR_OMEGA, MOUSE_FORCE, MOUSE_DENSITY, MOUSE_RADIUS,
    MOUSE_COLOR,
)
import kernels


class FluidEngine:
    """GPU-accelerated 2D fluid solver with RGB density transport."""

    def __init__(self, N=SIM_GRID_SIZE, use_sor=True):
        self.N = N
        self.size = N + 2  # 1-cell border padding on each side
        self.use_sor = use_sor

        # --- Allocate GPU memory (numba.cuda device arrays) ---

        # Velocity (scalar 2D)
        self.u  = cuda.device_array((self.size, self.size), dtype=np.float32)
        self.v  = cuda.device_array((self.size, self.size), dtype=np.float32)
        self.u0 = cuda.device_array_like(self.u)
        self.v0 = cuda.device_array_like(self.v)

        # Density (RGB 3D)
        self.d  = cuda.device_array((self.size, self.size, 3), dtype=np.float32)
        self.d0 = cuda.device_array_like(self.d)

        # Pressure solver
        self.p     = cuda.device_array_like(self.u)
        self.p_new = cuda.device_array_like(self.u)   # PRE-ALLOCATED (was per-frame)
        self.div   = cuda.device_array_like(self.u)

        # Vorticity
        self.curl = cuda.device_array_like(self.u)

        # Output image buffer  (N x N x 3, uint8)
        self.output = cuda.device_array((N, N, 3), dtype=np.uint8)

        # --- Kernel launch config ---
        self.tpb = (16, 16)
        self.bpg = ((self.size + 15) // 16, (self.size + 15) // 16)
        self.bpg_render = ((N + 15) // 16, (N + 15) // 16)

        # --- Warm up CUDA context (JIT compile kernels) ---
        self._compile_warmup()

    # ------------------------------------------------------------------
    #  One-time JIT compilation  (avoids stutter on first real frame)
    # ------------------------------------------------------------------
    def _compile_warmup(self):
        """Run a tiny step to trigger numba CUDA JIT compilation."""
        tmp = cuda.device_array((self.size, self.size), dtype=np.float32)
        kernels.k_fill[self.bpg, self.tpb](tmp, 0.0)
        cuda.synchronize()

    # ------------------------------------------------------------------
    #  Simulation step
    # ------------------------------------------------------------------
    def step(self, dt):
        # 1. Vorticity confinement (re-injects rotational energy)
        self._vorticity(dt)

        # 2. Velocity advection  (u -> u0,  v -> v0)
        kernels.k_advect[self.bpg, self.tpb](
            self.N, self.u0, self.u, self.u, self.v, dt, VELOCITY_DISSIPATION
        )
        kernels.k_advect[self.bpg, self.tpb](
            self.N, self.v0, self.v, self.u, self.v, dt, VELOCITY_DISSIPATION
        )
        # Swap current / previous
        self.u, self.u0 = self.u0, self.u
        self.v, self.v0 = self.v0, self.v

        kernels.k_set_boundary[self.bpg, self.tpb](self.N, 1, self.u)
        kernels.k_set_boundary[self.bpg, self.tpb](self.N, 2, self.v)

        # 3. Pressure projection (divergence-free)
        self._project()

        # 4. Density advection  (d -> d0)
        kernels.k_advect_rgb[self.bpg, self.tpb](
            self.N, self.d0, self.d, self.u, self.v, dt, DISSIPATION
        )
        self.d, self.d0 = self.d0, self.d

        kernels.k_set_boundary_rgb[self.bpg, self.tpb](self.N, self.d)

        # Clear the "previous" density buffer for next frame's sources
        kernels.k_fill[self.bpg, self.tpb](self.d0, 0.0)

    # ------------------------------------------------------------------
    #  Pressure projection  (SOR or Jacobi)
    # ------------------------------------------------------------------
    def _project(self):
        kernels.k_divergence[self.bpg, self.tpb](
            self.N, self.u, self.v, self.div, self.p
        )

        if self.use_sor:
            pressure_kernel = kernels.k_pressure_sor
            for _ in range(ITERATIONS):
                pressure_kernel[self.bpg, self.tpb](
                    self.N, self.p, self.div, self.p_new, SOR_OMEGA
                )
                # Swap pressure buffers
                self.p, self.p_new = self.p_new, self.p
        else:
            for _ in range(ITERATIONS):
                kernels.k_pressure_jacobi[self.bpg, self.tpb](
                    self.N, self.p, self.div, self.p_new
                )
                self.p, self.p_new = self.p_new, self.p

        kernels.k_gradient_subtract[self.bpg, self.tpb](
            self.N, self.u, self.v, self.p, self.u0, self.v0
        )
        self.u[:] = self.u0
        self.v[:] = self.v0

        kernels.k_set_boundary[self.bpg, self.tpb](self.N, 1, self.u)
        kernels.k_set_boundary[self.bpg, self.tpb](self.N, 2, self.v)

    # ------------------------------------------------------------------
    #  Vorticity confinement
    # ------------------------------------------------------------------
    def _vorticity(self, dt):
        kernels.k_vorticity_compute[self.bpg, self.tpb](
            self.N, self.u, self.v, self.curl
        )
        kernels.k_vorticity_apply[self.bpg, self.tpb](
            self.N, self.u, self.v, self.curl, dt, VORTICITY_STRENGTH
        )

    # ------------------------------------------------------------------
    #  Source injection (autonomous presets & mouse)
    # ------------------------------------------------------------------
    def add_source(self, x, y, amount, radius, vel_x, vel_y, color):
        """Inject RGB density + velocity at grid position (x, y)."""
        r, g, b = color
        kernels.k_add_source_rgb[self.bpg, self.tpb](
            self.N, self.d, self.u, self.v,
            float(x), float(y), float(radius),
            float(amount), float(vel_x), float(vel_y),
            float(r), float(g), float(b),
        )

    def add_mouse_source(self, gx, gy, prev_gx, prev_gy, color=None):
        """Inject fluid from mouse drag between two grid positions.

        Derives velocity from cursor movement for intuitive interaction.
        """
        if color is None:
            color = MOUSE_COLOR
        vx = (gx - prev_gx) * MOUSE_FORCE
        vy = (gy - prev_gy) * MOUSE_FORCE
        # Ensure some minimum force so still clicks also produce movement
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < 1.0:
            angle = ((gx + gy) * 0.1) % 6.2832
            vx = math.cos(angle) * MOUSE_FORCE * 2.0
            vy = math.sin(angle) * MOUSE_FORCE * 2.0
        self.add_source(
            gx, gy, MOUSE_DENSITY, MOUSE_RADIUS,
            vx, vy, color,
        )

    # ------------------------------------------------------------------
    #  Render to image
    # ------------------------------------------------------------------
    def render(self, sim_time, mode_idx):
        """Return (N, N, 3) uint8 numpy array."""
        kernels.k_render[self.bpg_render, self.tpb](
            self.N, self.d, self.u, self.v, self.output, sim_time, mode_idx
        )
        return self.output.copy_to_host()

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------
    def reset(self):
        for arr in (self.u, self.v, self.u0, self.v0):
            kernels.k_fill[self.bpg, self.tpb](arr, 0.0)
        for arr in (self.d, self.d0):
            kernels.k_fill[self.bpg, self.tpb](arr, 0.0)


