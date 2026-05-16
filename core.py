import numpy as np
from config import _R_MAP, _G_MAP, _B_MAP, PRESETS

# ---------------------------------------------------------------------------
#  Vectorised HSV → RGB
# ---------------------------------------------------------------------------
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
        
        # Velocity and Density fields
        self.u = np.zeros(s, np.float32) # Velocity X
        self.v = np.zeros(s, np.float32) # Velocity Y
        self.d = np.zeros(s3, np.float32) # Density (RGB)
        
        # Previous state buffers
        self.u0 = np.zeros(s, np.float32)
        self.v0 = np.zeros(s, np.float32)
        self.d0 = np.zeros(s3, np.float32)
        
        # Temporary buffers for solver
        self._tmp2a = np.zeros(s, np.float32)
        self._tmp2b = np.zeros(s, np.float32)
        self._tmp2c = np.zeros(s, np.float32)
        self._tmp3  = np.zeros(s3, np.float32)
        self._p     = np.zeros(s, np.float32)
        self._div   = np.zeros(s, np.float32)
        
        # Advection optimization
        ii, jj = np.meshgrid(
            np.arange(1, N+1, dtype=np.float32),
            np.arange(1, N+1, dtype=np.float32), 
            indexing="ij"
        )
        self._adv_ii = ii
        self._adv_jj = jj
        self._adv_dtN = self.dt * N
        
        # Brush mask for adding density/velocity
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
#  HDR Flow Renderer
# ---------------------------------------------------------------------------
class Renderer:
    def __init__(self, N):
        self.N = N
        self._rows = np.arange(N, dtype=np.int32)[:,None]
        self._cols = np.arange(N, dtype=np.int32)[None,:]
        
        # Reusable buffers
        self._hue  = np.zeros((N,N), np.float32)
        self._sat  = np.zeros((N,N), np.float32)
        self._val  = np.zeros((N,N), np.float32)
        self._stk  = np.zeros((4,N,N), np.float32)
        self._rgb  = np.zeros((N,N,3), np.float32)
        self._u8   = np.zeros((N,N,3), np.uint8)
        
        # Vignette mask
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
        np.sqrt(out, out=out) # Gamma correction
        np.multiply(out, 255.0, out=out)
        self._u8[:] = out
        
        return self._u8

    def _render_flow(self, density, u, v, t, psyche):
        N = self.N
        speed = np.sqrt(u*u + v*v) + 1e-5
        angle = np.arctan2(v, u)
        
        # Hue based on direction
        hue = (angle / (2.0 * np.pi)) + 0.5
        hue += t * 0.05 
        hue += density[:,:,0] * 0.05
        hue %= 1.0
        
        sat = np.clip(speed * 4.0 + 0.5, 0.6, 1.0)
        val = np.sum(density, axis=-1, out=self._val)
        
        # Make movement glow
        val += speed * 1.5 
        
        if psyche > 0.1:
            val += np.sin(t * 5.0) * 0.05 * psyche
            
        np.clip(val, 0.0, 1.5, out=val)

        hsv_to_rgb_vec(hue, sat, val, self._stk, self._rgb, self._rows, self._cols)
        
        # Bloom glow
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