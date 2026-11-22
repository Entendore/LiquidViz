import pygame
import numpy as np
import cv2
import sys
import math
from datetime import datetime

# Optional: Pillow for GIF export
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ========== CONFIGURATION ==========
WINDOW_W, WINDOW_H = 1280, 720
SIM_W, SIM_H = 480, 360
FPS = 60

# Physics Constants
FLOW_RATE = 0.25
GRAVITY = 0.18
VISCOSITY = 0.02
MAX_WATER = 1.0
MIN_WATER = 0.001

# ========== MINIMALIST PALETTES ==========
def palette_ink(v):
    i = int(v * 255)
    # FIX: Clamp blue channel to max 255
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

# ========== OPTIMIZED LIQUID ENGINE ==========
class LiquidEngine:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.water = np.zeros((h, w), dtype=np.float32)
        self.palette_idx = 0
        self._build_lut()
        
    def _build_lut(self):
        # Pre-calculate color Look-Up Table
        lut = []
        for i in range(256):
            v = i / 255.0
            # Get raw color
            r, g, b = PALETTES[self.palette_idx][1](v)
            # FIX: Safety clamp to ensure strictly 0-255
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
        num_drops = 50
        xs = np.random.randint(0, self.w, num_drops)
        ys = np.random.randint(0, self.h // 3, num_drops)
        for x, y in zip(xs, ys):
            self.water[y:y+5, x:x+5] = 0.6

    # OPTIMIZED PHYSICS
    def update(self):
        # 1. Gravity
        current = self.water[:-1, :]
        below = self.water[1:, :]
        diff = current - below
        flow = np.maximum(diff, 0) * FLOW_RATE
        self.water[:-1, :] -= flow
        self.water[1:, :] += flow
        
        # 2. Horizontal Dispersion (Convolution)
        kernel = np.array([[0, 0.1, 0],
                           [0.1, 0.6, 0.1],
                           [0, 0.1, 0]], dtype=np.float32)
        self.water = cv2.filter2D(self.water, -1, kernel)
        
        # 3. Damping & Clamping
        self.water *= (1.0 - VISCOSITY)
        np.clip(self.water, 0, MAX_WATER, out=self.water)

    def render(self, target_w, target_h):
        indices = np.clip((self.water * 255).astype(np.uint8), 0, 255)
        rgb_small = self.lut[indices]
        return cv2.resize(rgb_small, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

# ========== RECORDER ==========
class Recorder:
    def __init__(self):
        self.video_out = None
        self.gif_frames = []
        self.active = False
        self.format = 'mp4'
        
    def start(self, filename, fps, resolution, fmt='mp4'):
        if fmt == 'gif' and not PIL_AVAILABLE: fmt = 'mp4'
        
        if fmt == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_out = cv2.VideoWriter(filename, fourcc, fps, resolution)
        elif fmt == 'gif':
            self.gif_frames = []
            self.gif_name = filename
            
        self.active = True
        self.format = fmt
        print(f"Recording started: {filename}")

    def add_frame(self, frame):
        if not self.active: return
        if self.format == 'mp4':
            self.video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif self.format == 'gif':
            self.gif_frames.append(Image.fromarray(frame))

    def stop(self):
        if not self.active: return
        if self.format == 'mp4' and self.video_out:
            self.video_out.release()
        elif self.format == 'gif' and self.gif_frames:
            self.gif_frames[0].save(self.gif_name, save_all=True, append_images=self.gif_frames[1:], duration=20, loop=0)
        
        self.active = False
        print("Recording stopped.")

# ========== MAIN LOOP ==========
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Minimalist Liquid Simulation")
    clock = pygame.time.Clock()

    sim = LiquidEngine(SIM_W, SIM_H)
    sim.seed_solid()
    
    timer_palette = 0
    timer_preset = 0
    current_preset = 0
    
    PALETTE_INTERVAL = 6.0
    PRESET_INTERVAL = 10.0
    
    running = True
    
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Autonomous Logic
        timer_palette += dt
        timer_preset += dt
        
        if timer_palette > PALETTE_INTERVAL:
            timer_palette = 0
            sim.set_palette(sim.palette_idx + 1)
            
        if timer_preset > PRESET_INTERVAL:
            timer_preset = 0
            current_preset = (current_preset + 1) % 3
            if current_preset == 0: sim.seed_solid()
            elif current_preset == 1: sim.seed_wave()
            else: sim.seed_rain()

        # Physics
        for _ in range(3):
            sim.update()

        # Render
        screen.fill((10, 10, 15))
        frame = sim.render(WINDOW_W, WINDOW_H)
        
        # Correct array shape for Pygame (W, H, 3) -> (H, W, 3) is what render returns.
        # Pygame surfarray expects (W, H, 3), so we swap axes.
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        
        # Minimalist HUD
        font = pygame.font.SysFont("Arial", 18)
        title = font.render(f"Mode: {PALETTES[sim.palette_idx][0]}", True, (200, 200, 200))
        screen.blit(title, (20, 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()