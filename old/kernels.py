# kernels.py - CUDA kernels for the fluid simulation (numba.cuda)
import math
from numba import cuda, float32

# ---------------------------------------------------------------------------
#  Helper: Catmull-Rom cubic interpolation  (bicubic for sharp advection)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _cubic(x0, x1, x2, x3, t):
    """Catmull-Rom cubic interpolation — sharper than linear."""
    w0 = -0.5 * (t ** 3) + (t ** 2) - 0.5 * t
    w1 = 1.5 * (t ** 3) - 2.5 * (t ** 2) + 1.0
    w2 = -1.5 * (t ** 3) + 2.5 * (t ** 2) + 0.5 * t
    w3 = 0.5 * (t ** 3) - 0.5 * (t ** 2)
    return x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3


@cuda.jit(device=True)
def _sample_bicubic(arr, x, y, N):
    """Sample a 2D grid with bicubic interpolation, clamped to [1, N]."""
    i0 = int(x)
    j0 = int(y)
    fx = x - i0
    fy = y - j0

    st = cuda.local.array((4, 4), dtype=float32)
    for di in range(4):
        ci = min(max(i0 + di - 1, 1), N)
        for dj in range(4):
            cj = min(max(j0 + dj - 1, 1), N)
            st[di, dj] = arr[ci, cj]

    c0 = _cubic(st[0, 0], st[0, 1], st[0, 2], st[0, 3], fx)
    c1 = _cubic(st[1, 0], st[1, 1], st[1, 2], st[1, 3], fx)
    c2 = _cubic(st[2, 0], st[2, 1], st[2, 2], st[2, 3], fx)
    c3 = _cubic(st[3, 0], st[3, 1], st[3, 2], st[3, 3], fx)
    return _cubic(c0, c1, c2, c3, fy)


# ---------------------------------------------------------------------------
#  Utility kernels
# ---------------------------------------------------------------------------

@cuda.jit
def k_fill(arr, value):
    """Zero-out (or fill) a 2D or 3D device array."""
    i, j = cuda.grid(2)
    if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
        if arr.ndim == 3:
            arr[i, j, 0] = value
            arr[i, j, 1] = value
            arr[i, j, 2] = value
        else:
            arr[i, j] = value


# ---------------------------------------------------------------------------
#  Advection  (bicubic – sharp fluid details)
# ---------------------------------------------------------------------------

@cuda.jit
def k_advect(N, d_out, d_in, u, v, dt, dissipation):
    """Semi-Lagrangian advection for a scalar 2D field."""
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        dt0 = dt * N
        x = max(1.5, min(N + 0.5, i - dt0 * u[i, j]))
        y = max(1.5, min(N + 0.5, j - dt0 * v[i, j]))
        d_out[i, j] = dissipation * _sample_bicubic(d_in, x, y, N)


@cuda.jit
def k_advect_rgb(N, d_out, d_in, u, v, dt, dissipation):
    """Semi-Lagrangian advection for an RGB 3D density field."""
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        dt0 = dt * N
        x = max(1.5, min(N + 0.5, i - dt0 * u[i, j]))
        y = max(1.5, min(N + 0.5, j - dt0 * v[i, j]))

        i0 = int(x)
        j0 = int(y)
        fx = x - i0
        fy = y - j0

        for k in range(3):
            st = cuda.local.array((4, 4), dtype=float32)
            for di in range(4):
                ci = min(max(i0 + di - 1, 1), N)
                for dj in range(4):
                    cj = min(max(j0 + dj - 1, 1), N)
                    st[di, dj] = d_in[ci, cj, k]

            c0 = _cubic(st[0, 0], st[0, 1], st[0, 2], st[0, 3], fx)
            c1 = _cubic(st[1, 0], st[1, 1], st[1, 2], st[1, 3], fx)
            c2 = _cubic(st[2, 0], st[2, 1], st[2, 2], st[2, 3], fx)
            c3 = _cubic(st[3, 0], st[3, 1], st[3, 2], st[3, 3], fx)
            val = _cubic(c0, c1, c2, c3, fy)
            d_out[i, j, k] = dissipation * val


# ---------------------------------------------------------------------------
#  Boundary conditions
# ---------------------------------------------------------------------------

@cuda.jit
def k_set_boundary(N, b, x):
    """Apply boundary conditions for velocity or scalar fields."""
    i, j = cuda.grid(2)
    s = N + 1  # border index
    if i == 0 or i == s:
        if 1 <= j <= N:
            src = i + 1 if i == 0 else i - 1
            if b == 1:
                x[i, j] = -x[src, j]
            else:
                x[i, j] = x[src, j]
    if j == 0 or j == s:
        if 1 <= i <= N:
            src = j + 1 if j == 0 else j - 1
            if b == 2:
                x[i, j] = -x[i, src]
            else:
                x[i, j] = x[i, src]
    # Corners: average of neighbours
    if i == 0 and j == 0:
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    elif i == 0 and j == s:
        x[0, s] = 0.5 * (x[1, s] + x[0, N])
    elif i == s and j == 0:
        x[s, 0] = 0.5 * (x[N, 0] + x[s, 1])
    elif i == s and j == s:
        x[s, s] = 0.5 * (x[N, s] + x[s, N])


@cuda.jit
def k_set_boundary_rgb(N, d):
    """Boundary conditions for RGB density (Neumann – zero gradient)."""
    i, j = cuda.grid(2)
    s = N + 1
    if i == 0 or i == s:
        if 1 <= j <= N:
            src = i + 1 if i == 0 else i - 1
            d[i, j, 0] = d[src, j, 0]
            d[i, j, 1] = d[src, j, 1]
            d[i, j, 2] = d[src, j, 2]
    if j == 0 or j == s:
        if 1 <= i <= N:
            src = j + 1 if j == 0 else j - 1
            d[i, src, 0]  # just read to trigger boundary
            d[i, j, 0] = d[i, src, 0]
            d[i, j, 1] = d[i, src, 1]
            d[i, j, 2] = d[i, src, 2]


# ---------------------------------------------------------------------------
#  Pressure projection
# ---------------------------------------------------------------------------

@cuda.jit
def k_divergence(N, u, v, div, p):
    """Compute divergence of velocity field; zero-out pressure."""
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        h = 1.0 / N
        div[i, j] = -0.5 * h * (
            (u[i + 1, j] - u[i - 1, j]) + (v[i, j + 1] - v[i, j - 1])
        )
        p[i, j] = 0.0


@cuda.jit
def k_pressure_sor(N, p, div, p_new, omega):
    """SOR (Successive Over-Relaxation) Jacobi iteration for pressure.

    Converges significantly faster than plain Jacobi when omega > 1.0.
    Typical good values: 1.5 – 1.9.
    """
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        neighbor_avg = (
            p[i - 1, j] + p[i + 1, j] + p[i, j - 1] + p[i, j + 1]
        ) / 4.0
        p_jacobi = div[i, j] + neighbor_avg
        # SOR: blend between old and new
        p_new[i, j] = (1.0 - omega) * p[i, j] + omega * p_jacobi


@cuda.jit
def k_pressure_jacobi(N, p, div, p_new):
    """Plain Jacobi iteration (kept as fallback)."""
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        p_new[i, j] = (
            div[i, j]
            + p[i - 1, j] + p[i + 1, j]
            + p[i, j - 1] + p[i, j + 1]
        ) / 4.0


@cuda.jit
def k_gradient_subtract(N, u, v, p, u_new, v_new):
    """Subtract pressure gradient to make velocity divergence-free."""
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        h = 1.0 / N
        u_new[i, j] = u[i, j] - 0.5 * (p[i + 1, j] - p[i - 1, j]) / h
        v_new[i, j] = v[i, j] - 0.5 * (p[i, j + 1] - p[i, j - 1]) / h


# ---------------------------------------------------------------------------
#  Vorticity confinement
# ---------------------------------------------------------------------------

@cuda.jit
def k_vorticity_compute(N, u, v, curl):
    """Compute curl (vorticity scalar) of the 2D velocity field."""
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        curl[i, j] = (
            (v[i + 1, j] - v[i - 1, j])
            - (u[i, j + 1] - u[i, j - 1])
        ) * 0.5


@cuda.jit
def k_vorticity_apply(N, u, v, curl, dt, strength):
    """Apply vorticity confinement force to counteract numerical diffusion."""
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        # Gradient of |curl|
        d_abs_dx = (abs(curl[i + 1, j]) - abs(curl[i - 1, j])) * 0.5
        d_abs_dy = (abs(curl[i, j + 1]) - abs(curl[i, j - 1])) * 0.5

        len_sq = d_abs_dx * d_abs_dx + d_abs_dy * d_abs_dy + 1e-5
        inv_len = 1.0 / math.sqrt(len_sq)
        nx = d_abs_dx * inv_len
        ny = d_abs_dy * inv_len

        force = curl[i, j] * strength
        u[i, j] += dt * force * nx
        v[i, j] -= dt * force * ny


# ---------------------------------------------------------------------------
#  Source injection
# ---------------------------------------------------------------------------

@cuda.jit
def k_add_source_rgb(N, d, u, v, cx, cy, radius, amount, fx, fy, r, g, b):
    """Inject density (RGB) and velocity at a point with Gaussian falloff."""
    i, j = cuda.grid(2)
    dx = i - cx
    dy = j - cy
    dist_sq = dx * dx + dy * dy
    rad_sq = radius * radius

    if dist_sq < rad_sq * 4.0:  # 2-sigma cutoff for efficiency
        falloff = math.exp(-dist_sq / (rad_sq * 0.4))

        d[i, j, 0] += amount * r * falloff
        d[i, j, 1] += amount * g * falloff
        d[i, j, 2] += amount * b * falloff

        u[i, j] += fx * falloff
        v[i, j] += fy * falloff


# ---------------------------------------------------------------------------
#  Render kernel
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _hsv_to_rgb(h, s, v):
    """Per-pixel HSV to RGB (used inside render kernels)."""
    if s == 0.0:
        return v, v, v
    h6 = h * 6.0
    i = int(h6) % 6
    f = h6 - int(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q


@cuda.jit
def k_render(N, density, u, v, out_img, sim_time, mode):
    """Map fluid state to an RGB uint8 image.

    Modes:
        0  Vibrant Oil  – saturation-boosted density with directional hue shift
        1  Neon Glow    – bright HDR tonemapped density
        2  Velocity Field – u/v mapped to R/G channels
    """
    i, j = cuda.grid(2)
    if 0 <= i < N and 0 <= j < N:
        gi, gj = i + 1, j + 1

        d_r = density[gi, gj, 0]
        d_g = density[gi, gj, 1]
        d_b = density[gi, gj, 2]

        # Unsharp-mask sharpening (local contrast boost)
        avg_r = (density[gi - 1, gj, 0] + density[gi + 1, gj, 0]
                 + density[gi, gj - 1, 0] + density[gi, gj + 1, 0]) * 0.25
        avg_g = (density[gi - 1, gj, 1] + density[gi + 1, gj, 1]
                 + density[gi, gj - 1, 1] + density[gi, gj + 1, 1]) * 0.25
        avg_b = (density[gi - 1, gj, 2] + density[gi + 1, gj, 2]
                 + density[gi, gj - 1, 2] + density[gi, gj + 1, 2]) * 0.25

        sharp = 0.3
        d_r = d_r + (d_r - avg_r) * sharp
        d_g = d_g + (d_g - avg_g) * sharp
        d_b = d_b + (d_b - avg_b) * sharp

        r, g, b = 0.0, 0.0, 0.0

        if mode == 0:
            # --- Vibrant Oil ---
            r, g, b = d_r, d_g, d_b
            avg = (r + g + b) / 3.0
            r = r + (r - avg) * 1.5
            g = g + (g - avg) * 1.5
            b = b + (b - avg) * 1.5

            # Reinhard tonemapping
            r = r / (1.0 + r)
            g = g / (1.0 + g)
            b = b / (1.0 + b)

            # Speed-based specular highlight
            spd = math.sqrt(u[gi, gj] * u[gi, gj] + v[gi, gj] * v[gi, gj])
            shine = min(1.0, spd * 0.4)
            r += shine * 0.15
            g += shine * 0.15
            b += shine * 0.2

        elif mode == 1:
            # --- Neon Glow ---
            # Subtle time-based hue rotation for living feel
            hue_shift = sim_time * 0.03
            angle = math.atan2(d_b - d_g, d_r - d_g + 1e-5)
            hue = (angle / 6.2832 + 0.5 + hue_shift) % 1.0
            lum = (d_r + d_g + d_b) / 3.0
            sat = min(1.0, lum * 2.0 + 0.3)

            r, g, b = _hsv_to_rgb(hue, sat, min(1.0, lum * 1.8))

        else:
            # --- Velocity Field ---
            r = abs(u[gi, gj]) * 3.0
            g = abs(v[gi, gj]) * 3.0
            b = 0.1
            r = r / (1.0 + r)
            g = g / (1.0 + g)
            b = b / (1.0 + b)

        # Vignette
        dx = (i - N * 0.5) / N
        dy = (j - N * 0.5) / N
        dist = math.sqrt(dx * dx + dy * dy)
        vig = max(0.0, 1.0 - dist * 0.35)

        out_img[i, j, 0] = min(255, max(0, int(r * vig * 255.0)))
        out_img[i, j, 1] = min(255, max(0, int(g * vig * 255.0)))
        out_img[i, j, 2] = min(255, max(0, int(b * vig * 255.0)))