import numpy as np

# Simulation Resolution (Internal)
SIM_W, SIM_H = 480, 360
FPS = 60

# Physics Constants (Shallow Water Equations)
SPEED_OF_WAVE = 0.5     # How fast ripples travel
DAMPING = 0.995         # How much energy is kept (1.0 = infinite)
GRAVITY = 0.1           # Downward pull
RAIN_RATE = 0.02        # Chance of rain drop per frame

# Export Resolutions
RES_STANDARD = (1920, 1080)
RES_SHORTS = (1080, 1920)

# ========== VIBRANT PALETTES ==========
def palette_lava(v):
    # Hot orange/red
    r = int(255 * min(1.0, v * 1.5))
    g = int(100 * max(0, v - 0.2))
    b = int(50 * max(0, v - 0.5))
    return (r, g, b)

def palette_ocean(v):
    # Deep blue/cyan
    r = int(20 + 60 * v)
    g = int(80 + 175 * v)
    b = int(150 + 105 * min(1, v * 1.5))
    return (r, g, b)

def palette_neon(v):
    # Cyberpunk pink/cyan
    r = int(255 * v)
    g = int(50 * (1 - abs(v - 0.5)*2)) # Peak green in middle
    b = int(255 * (1 - v))
    return (r, g, b)

def palette_toxic(v):
    # Radioactive green
    r = int(50 + 150 * v)
    g = int(200 + 55 * v)
    b = int(50 + 100 * (1-v))
    return (r, g, b)

def palette_royal(v):
    # Deep purple/gold
    r = int(80 + 175 * v)
    g = int(50 + 100 * (v ** 2))
    b = int(200 * (1 - v * 0.5))
    return (r, g, b)

PALETTES = [
    ("Lava", palette_lava),
    ("Ocean", palette_ocean),
    ("Neon", palette_neon),
    ("Toxic", palette_toxic),
    ("Royal", palette_royal)
]