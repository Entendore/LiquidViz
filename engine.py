# engine.py
import cupy as cp
import numpy as np
from config import SIM_GRID_SIZE, ITERATIONS, DISSIPATION, VELOCITY_DISSIPATION, VORTICITY_STRENGTH
import kernels

class FluidEngine:
    def __init__(self, N=SIM_GRID_SIZE):
        self.N = N
        self.size = N + 2  # Include borders (1-cell padding)
        
        # --- Allocate GPU Memory ---
        
        # Velocity Fields (Scalar 2D)
        self.u = cp.zeros((self.size, self.size), dtype=cp.float32)
        self.v = cp.zeros((self.size, self.size), dtype=cp.float32)
        self.u0 = cp.zeros_like(self.u)
        self.v0 = cp.zeros_like(self.v)
        
        # Density Field (RGB 3D)
        self.d = cp.zeros((self.size, self.size, 3), dtype=cp.float32)
        self.d0 = cp.zeros_like(self.d)
        
        # Pressure Solver Buffers
        self.p = cp.zeros_like(self.u)
        self.div = cp.zeros_like(self.u)
        
        # Vorticity Buffer
        self.curl = cp.zeros_like(self.u)
        
        # Output Image Buffer (N x N x 3 uint8)
        self.output = cp.zeros((N, N, 3), dtype=cp.uint8)
        
        # --- Kernel Launch Configuration ---
        self.tpb = (16, 16)  # Threads per block
        self.bpg = ((self.size + 15) // 16, (self.size + 15) // 16)
        self.bpg_render = ((N + 15) // 16, (N + 15) // 16)

    def step(self, dt):
        # 1. Vorticity Confinement (injects energy into u/v directly)
        self._vorticity(dt)
        
        # 2. Velocity Step
        # Advect velocity: Read from u (current), Write to u0 (next)
        kernels.k_advect[self.bpg, self.tpb](
            self.N, self.u0, self.u, self.u, self.v, dt, VELOCITY_DISSIPATION
        )
        kernels.k_advect[self.bpg, self.tpb](
            self.N, self.v0, self.v, self.u, self.v, dt, VELOCITY_DISSIPATION
        )
        
        # Swap: Move new velocity (u0/v0) to current (u/v), clear u0/v0 implicitly for next use
        self.u, self.u0 = self.u0, self.u
        self.v, self.v0 = self.v0, self.v
        
        kernels.k_set_boundary[self.bpg, self.tpb](self.N, 1, self.u)
        kernels.k_set_boundary[self.bpg, self.tpb](self.N, 2, self.v)
        
        # 3. Projection (Make velocity divergence-free)
        self._project()
        
        # 4. Density Step
        # Advect density: Read from d (current), Write to d0 (next)
        kernels.k_advect_rgb[self.bpg, self.tpb](
            self.N, self.d0, self.d, self.u, self.v, dt, DISSIPATION
        )
        
        # Swap: Move new density (d0) to current (d)
        self.d, self.d0 = self.d0, self.d
        
        # Apply Boundary Conditions
        kernels.k_set_boundary_rgb[self.bpg, self.tpb](self.N, self.d)
        
        # Clear the 'previous' buffer (which is now d0 after swap) for the next frame
        # (Optional but good hygiene, ensures no ghosting if logic changes)
        self.d0.fill(0)

    def _project(self):
        kernels.k_divergence[self.bpg, self.tpb](
            self.N, self.u, self.v, self.div, self.p
        )
        
        p_new = cp.zeros_like(self.p)
        for _ in range(ITERATIONS):
            kernels.k_pressure_jacobi[self.bpg, self.tpb](
                self.N, self.p, self.div, p_new
            )
            self.p[:] = p_new
        
        kernels.k_gradient_subtract[self.bpg, self.tpb](
            self.N, self.u, self.v, self.p, self.u0, self.v0
        )
        
        self.u[:] = self.u0
        self.v[:] = self.v0
        
        kernels.k_set_boundary[self.bpg, self.tpb](self.N, 1, self.u)
        kernels.k_set_boundary[self.bpg, self.tpb](self.N, 2, self.v)

    def _vorticity(self, dt):
        kernels.k_vorticity_compute[self.bpg, self.tpb](
            self.N, self.u, self.v, self.curl
        )
        kernels.k_vorticity_apply[self.bpg, self.tpb](
            self.N, self.u, self.v, self.curl, dt, VORTICITY_STRENGTH
        )

    def add_source(self, x, y, amount, radius, vel_x, vel_y, color):
        """
        Adds density (RGB) and velocity at a specific point.
        IMPORTANT: Writes to 'd', 'u', 'v' (current state buffers).
        """
        r, g, b = color
        
        kernels.k_add_source_rgb[self.bpg, self.tpb](
            self.N, self.d, self.u, self.v, 
            float(x), float(y), float(radius), 
            float(amount), float(vel_x), float(vel_y),
            float(r), float(g), float(b)
        )

    def render(self, time, mode_idx):
        kernels.k_render[self.bpg_render, self.tpb](
            self.N, self.d, self.u, self.v, self.output, time, mode_idx
        )
        return cp.asnumpy(self.output)

    def reset(self):
        self.u.fill(0)
        self.v.fill(0)
        self.d.fill(0)
        self.u0.fill(0)
        self.v0.fill(0)
        self.d0.fill(0)