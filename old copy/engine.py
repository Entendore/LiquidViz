import cupy as cp
import numpy as np
from config import SIM_W, SIM_H, PALETTES, SPEED_OF_WAVE, DAMPING, RAIN_RATE

class LiquidEngine:
    def __init__(self):
        # Dimensions (renamed to avoid conflict with height map)
        self.width, self.height = SIM_W, SIM_H
        
        # State:
        # h_map: Height map (amount of water)
        # u, v: Velocity fields
        self.h_map = cp.zeros((self.height, self.width), dtype=np.float32)
        self.u = cp.zeros((self.height, self.width), dtype=np.float32)
        self.v = cp.zeros((self.height, self.width), dtype=np.float32)
        
        self.palette_idx = 0
        self._build_lut()
        
    def _build_lut(self):
        # Build color LUT on CPU, upload to GPU
        lut = []
        for i in range(256):
            v = i / 255.0
            r, g, b = PALETTES[self.palette_idx][1](v)
            lut.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        self.lut = cp.array(lut, dtype=np.uint8)
    
    def set_palette(self, idx):
        self.palette_idx = idx % len(PALETTES)
        self._build_lut()

    def seed_solid(self):
        # Clear and create a big block of water
        self.h_map.fill(0)
        self.u.fill(0)
        self.v.fill(0)
        cx, cy = self.width // 2, self.height // 2
        self.h_map[cy-40:cy+40, cx-60:cx+60] = 1.0

    def seed_wave(self):
        self.h_map.fill(0)
        # Create a line of water
        self.h_map[self.height//2-10:self.height//2+10, :] = 0.8

    def seed_rain(self):
        # Continuous rain handled in update
        pass

    def update(self):
        # 1. Add Random Rain (continuous interaction)
        if cp.random.random() < RAIN_RATE:
            rx = cp.random.randint(0, self.width)
            ry = cp.random.randint(0, self.height)
            self.h_map[ry, rx] += 0.5
        
        # 2. Compute Gradients (Slopes)
        # Using roll to look at neighbors efficiently
        h_left = cp.roll(self.h_map, 1, axis=1)
        h_right = cp.roll(self.h_map, -1, axis=1)
        h_up = cp.roll(self.h_map, 1, axis=0)
        h_down = cp.roll(self.h_map, -1, axis=0)

        # 3. Update Velocity (Shallow Water Equations)
        # Flow from high to low
        self.u -= SPEED_OF_WAVE * (h_right - h_left) / 2.0
        self.v -= SPEED_OF_WAVE * (h_down - h_up) / 2.0
        
        # 4. Update Height based on Velocity divergence
        # If water flows out, height drops. If flows in, height rises.
        u_diff = (cp.roll(self.u, -1, axis=1) - cp.roll(self.u, 1, axis=1)) / 2.0
        v_diff = (cp.roll(self.v, -1, axis=0) - cp.roll(self.v, 1, axis=0)) / 2.0
        
        self.h_map += (u_diff + v_diff)

        # 5. Damping (Viscosity)
        self.h_map *= DAMPING
        self.u *= DAMPING
        self.v *= DAMPING
        
        # Clamp to prevent explosion
        self.h_map = cp.clip(self.h_map, 0, 2.0)

    def render(self):
        """
        Renders the height map with simple lighting.
        """
        # 1. Calculate Surface Normals for Lighting
        # Approximate gradient (slope)
        dx = cp.roll(self.h_map, -1, axis=1) - cp.roll(self.h_map, 1, axis=1)
        dy = cp.roll(self.h_map, -1, axis=0) - cp.roll(self.h_map, 1, axis=0)
        
        # Simple fake shading: 
        # If facing 'right' and 'up' (light source top-right), it's brighter.
        # dx positive = slope facing right = light hits it
        shade = (dx * 0.5) + (dy * 0.5) + 0.5
        
        # 2. Combine Height and Shade for final value
        # We use height for the color base, shade for brightness
        brightness = self.h_map + shade * 0.5
        
        # Normalize to 0-255 for LUT
        indices = cp.clip(brightness * 128, 0, 255).astype(cp.uint8)
        
        # 3. Map to Colors
        rgb_gpu = self.lut[indices]
        
        # 4. Move to CPU
        return cp.asnumpy(rgb_gpu)