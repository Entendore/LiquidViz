# kernels.py
import numpy as np
from numba import cuda, float32
import math

# ---------------------------------------------------------
# Helper: Cubic Interpolation Kernel (BSpline)
# ---------------------------------------------------------

@cuda.jit(device=True)
def cubic_interp(x0, x1, x2, x3, t):
    """
    Catmull-Rom cubic interpolation for sharp fluid details.
    Prevents the 'blur' associated with standard linear advection.
    """
    # Weights for Catmull-Rom
    # -0.5 * t^3 + t^2 - 0.5 * t
    w0 = -0.5 * (t**3) + (t**2) - 0.5 * t
    # 1.5 * t^3 - 2.5 * t^2 + 1
    w1 =  1.5 * (t**3) - 2.5 * (t**2) + 1.0
    # -1.5 * t^3 + 2.5 * t^2 - 0.5 * t
    w2 = -1.5 * (t**3) + 2.5 * (t**2) + 0.5 * t
    # 0.5 * t^3 - 0.5 * t^2
    w3 =  0.5 * (t**3) - 0.5 * (t**2)
    
    return x0*w0 + x1*w1 + x2*w2 + x3*w3

@cuda.jit(device=True)
def sample_cubic_2d(arr, x, y, N):
    """
    Samples a 2D grid using bicubic interpolation.
    Handles boundary clamping automatically.
    """
    # Grid coordinates
    i = int(x)
    j = int(y)
    
    # Fractional part
    fx = x - i
    fy = y - j
    
    # Clamp to valid range (need 1 extra pixel border for cubic)
    # We sample i-1 to i+2
    
    # Fetch 4x4 stencils
    # Weights require careful clamping
    vals = cuda.local.array((4,4), dtype=float32)
    
    for di in range(4):
        for dj in range(4):
            ci = min(max(i + di - 1, 1), N) # Clamp X
            cj = min(max(j + dj - 1, 1), N) # Clamp Y
            vals[di, dj] = arr[ci, cj]
            
    # Interpolate columns (X direction)
    col0 = cubic_interp(vals[0,0], vals[0,1], vals[0,2], vals[0,3], fx)
    col1 = cubic_interp(vals[1,0], vals[1,1], vals[1,2], vals[1,3], fx)
    col2 = cubic_interp(vals[2,0], vals[2,1], vals[2,2], vals[2,3], fx)
    col3 = cubic_interp(vals[3,0], vals[3,1], vals[3,2], vals[3,3], fx)
    
    # Interpolate rows (Y direction)
    return cubic_interp(col0, col1, col2, col3, fy)

# ---------------------------------------------------------
# Physics Kernels
# ---------------------------------------------------------

@cuda.jit
def k_advect(N, d, d0, u, v, dt, dissipation):
    """
    Advection using High-Quality Cubic Interpolation.
    """
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        dt0 = dt * N
        
        # Trace back
        x = i - dt0 * u[i, j]
        y = j - dt0 * v[i, j]
        
        # Clamp to grid boundaries
        x = max(1.5, min(N + 0.5, x))
        y = max(1.5, min(N + 0.5, y))
        
        # Sample using Cubic Interpolation
        val = sample_cubic_2d(d0, x, y, N)
        
        d[i, j] = dissipation * val

@cuda.jit
def k_advect_rgb(N, d, d0, u, v, dt, dissipation):
    """
    Advection for RGB using High-Quality Cubic Interpolation.
    This preserves color boundaries and prevents blurring.
    """
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        dt0 = dt * N
        
        x = i - dt0 * u[i, j]
        y = j - dt0 * v[i, j]
        
        x = max(1.5, min(N + 0.5, x))
        y = max(1.5, min(N + 0.5, y))
        
        # Interpolate each color channel separately
        for k in range(3):
            # We need a small helper to sample the 3D array, 
            # reimplementing the logic slightly for efficiency
            i0 = int(x)
            j0 = int(y)
            fx = x - i0
            fy = y - j0
            
            # Fetch 4x4 stencil for channel k
            # Local array for stencil values
            st = cuda.local.array((4,4), dtype=float32)
            
            # Unrolled sampling for performance
            # Indices: i0-1 to i0+2, j0-1 to j0+2
            # We clamp indices to grid
            for di in range(4):
                ci = min(max(i0 + di - 1, 1), N)
                for dj in range(4):
                    cj = min(max(j0 + dj - 1, 1), N)
                    st[di, dj] = d0[ci, cj, k]
            
            # Bicubic
            c0 = cubic_interp(st[0,0], st[0,1], st[0,2], st[0,3], fx)
            c1 = cubic_interp(st[1,0], st[1,1], st[1,2], st[1,3], fx)
            c2 = cubic_interp(st[2,0], st[2,1], st[2,2], st[2,3], fx)
            c3 = cubic_interp(st[3,0], st[3,1], st[3,2], st[3,3], fx)
            
            val = cubic_interp(c0, c1, c2, c3, fy)
            d[i, j, k] = dissipation * val

@cuda.jit
def k_set_boundary(N, b, x):
    i, j = cuda.grid(2)
    if i == 0 or i == N + 1:
        if 1 <= j <= N:
            if b == 1: x[i, j] = -x[i+1 if i==0 else i-1, j]
            else:       x[i, j] = x[i+1 if i==0 else i-1, j]
    if j == 0 or j == N + 1:
        if 1 <= i <= N:
            if b == 2: x[i, j] = -x[i, j+1 if j==0 else j-1]
            else:       x[i, j] = x[i, j+1 if j==0 else j-1]
    if i == 0 and j == 0:     x[i, j] = 0.5 * (x[1, 0] + x[0, 1])
    if i == 0 and j == N + 1: x[i, j] = 0.5 * (x[1, N+1] + x[0, N])
    if i == N + 1 and j == 0: x[i, j] = 0.5 * (x[N, 0] + x[N+1, 1])
    if i == N + 1 and j == N + 1: x[i, j] = 0.5 * (x[N, N+1] + x[N+1, N])

@cuda.jit
def k_set_boundary_rgb(N, d):
    i, j = cuda.grid(2)
    if i == 0 or i == N + 1:
        if 1 <= j <= N:
            src_i = i+1 if i==0 else i-1
            d[i, j, 0] = d[src_i, j, 0]
            d[i, j, 1] = d[src_i, j, 1]
            d[i, j, 2] = d[src_i, j, 2]
    if j == 0 or j == N + 1:
        if 1 <= i <= N:
            src_j = j+1 if j==0 else j-1
            d[i, j, 0] = d[i, src_j, 0]
            d[i, j, 1] = d[i, src_j, 1]
            d[i, j, 2] = d[i, src_j, 2]

@cuda.jit
def k_divergence(N, u, v, div, p):
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        h = 1.0 / N
        div[i, j] = -0.5 * h * ((u[i+1, j] - u[i-1, j]) + (v[i, j+1] - v[i, j-1]))
        p[i, j] = 0.0

@cuda.jit
def k_pressure_jacobi(N, p, div, p_new):
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        p_new[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / 4.0

@cuda.jit
def k_gradient_subtract(N, u, v, p, u_new, v_new):
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        h = 1.0 / N
        u_new[i, j] = u[i, j] - 0.5 * (p[i+1, j] - p[i-1, j]) / h
        v_new[i, j] = v[i, j] - 0.5 * (p[i, j+1] - p[i, j-1]) / h

@cuda.jit
def k_vorticity_compute(N, u, v, curl):
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        curl[i, j] = (v[i+1, j] - v[i-1, j] - u[i, j+1] + u[i, j-1]) * 0.5

@cuda.jit
def k_vorticity_apply(N, u, v, curl, dt, strength):
    i, j = cuda.grid(2)
    if 1 <= i <= N and 1 <= j <= N:
        # Compute gradient of |curl|
        d_abs_dx = (abs(curl[i+1, j]) - abs(curl[i-1, j])) * 0.5
        d_abs_dy = (abs(curl[i, j+1]) - abs(curl[i, j-1])) * 0.5
        
        len_sq = d_abs_dx*d_abs_dx + d_abs_dy*d_abs_dy + 1e-5
        nx = d_abs_dx / math.sqrt(len_sq)
        ny = d_abs_dy / math.sqrt(len_sq)
        
        # Apply force
        force = curl[i, j] * strength
        u[i, j] += dt * force * nx
        v[i, j] -= dt * force * ny

@cuda.jit
def k_add_source_rgb(N, d, u, v, cx, cy, radius, amount, force_x, force_y, r, g, b):
    i, j = cuda.grid(2)
    
    dx = i - cx
    dy = j - cy
    dist_sq = dx*dx + dy*dy
    rad_sq = radius * radius
    
    if dist_sq < rad_sq:
        dist = math.sqrt(dist_sq)
        falloff = math.exp(-dist_sq / (rad_sq * 0.4))
        
        d[i, j, 0] += amount * r * falloff
        d[i, j, 1] += amount * g * falloff
        d[i, j, 2] += amount * b * falloff
        
        u[i, j] += force_x * falloff
        v[i, j] += force_y * falloff

# ---------------------------------------------------------
# Render Kernel - Sharpened Output
# ---------------------------------------------------------

@cuda.jit(device=True)
def hsv_to_rgb_device(h, s, v):
    if s == 0.0: return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q

@cuda.jit
def k_render(N, density, u, v, out_img, time, mode):
    i, j = cuda.grid(2)
    if 0 <= i < N and 0 <= j < N:
        gi, gj = i + 1, j + 1
        
        d_r = density[gi, gj, 0]
        d_g = density[gi, gj, 1]
        d_b = density[gi, gj, 2]
        
        r, g, b = 0.0, 0.0, 0.0
        
        # Sharpening logic (Unsharp Mask approximation)
        # Sample center vs average of neighbors to boost local contrast
        avg_r = (density[gi-1,gj,0] + density[gi+1,gj,0] + density[gi,gj-1,0] + density[gi,gj+1,0]) * 0.25
        avg_g = (density[gi-1,gj,1] + density[gi+1,gj,1] + density[gi,gj-1,1] + density[gi,gj+1,1]) * 0.25
        avg_b = (density[gi-1,gj,2] + density[gi+1,gj,2] + density[gi,gj-1,2] + density[gi,gj+1,2]) * 0.25
        
        # Boost difference
        sharp_strength = 0.3
        d_r = d_r + (d_r - avg_r) * sharp_strength
        d_g = d_g + (d_g - avg_g) * sharp_strength
        d_b = d_b + (d_b - avg_b) * sharp_strength

        if mode == 0: # Vibrant Oil
            r, g, b = d_r, d_g, d_b
            avg = (r + g + b) / 3.0
            r = r + (r - avg) * 1.5
            g = g + (g - avg) * 1.5
            b = b + (b - avg) * 1.5
            
            r = r / (1.0 + r)
            g = g / (1.0 + g)
            b = b / (1.0 + b)
            
            speed = math.sqrt(u[gi,gj]*u[gi,gj] + v[gi,gj]*v[gi,gj])
            shine = min(1.0, speed * 0.4)
            r += shine * 0.15
            g += shine * 0.15
            b += shine * 0.2
            
        elif mode == 1: # Neon Glow
            r = d_r * 1.5
            g = d_g * 1.5
            b = d_b * 1.5
            r = r / (1.0 + r)
            g = g / (1.0 + g)
            b = b / (1.0 + b)
            
        else: # Velocity Field
            r = abs(u[gi, gj]) * 3.0
            g = abs(v[gi, gj]) * 3.0
            b = 0.1
            r = r / (1.0 + r)
            g = g / (1.0 + g)
            b = b / (1.0 + b)

        dist = math.sqrt(((i - N/2.0)/N)**2 + ((j - N/2.0)/N)**2)
        vig = 1.0 - dist * 0.3
        vig = max(0.0, vig)
        
        out_img[i, j, 0] = int(min(255.0, max(0.0, r * vig * 255.0)))
        out_img[i, j, 1] = int(min(255.0, max(0.0, g * vig * 255.0)))
        out_img[i, j, 2] = int(min(255.0, max(0.0, b * vig * 255.0)))