# config.py - Central configuration for LiquidViz fluid simulation
import numpy as np

# ---------------------------------------------------------------------------
#  Simulation Grid
# ---------------------------------------------------------------------------
SIM_GRID_SIZE = 512
FPS = 60

# ---------------------------------------------------------------------------
#  Physics Constants
# ---------------------------------------------------------------------------
ITERATIONS = 20                # Jacobi/SOR pressure iterations (was 12)
DISSIPATION = 0.998            # Density decay per step (was 0.999)
VELOCITY_DISSIPATION = 0.995   # Velocity decay per step (was 0.99)
VORTICITY_STRENGTH = 30.0      # Vorticity confinement strength
SOR_OMEGA = 1.7                # SOR relaxation factor for pressure solve (NEW)

# ---------------------------------------------------------------------------
#  Visualization Modes  (index must match kernel mode parameter)
# ---------------------------------------------------------------------------
RENDER_MODES = ["Vibrant Oil", "Neon Glow", "Velocity Field"]

# ---------------------------------------------------------------------------
#  HSV -> RGB vectorised lookup tables  (used by core.py CPU renderer)
# ---------------------------------------------------------------------------
#   stack[0] = v,  stack[1] = q,  stack[2] = p,  stack[3] = t
#   where p=v(1-s), q=v(1-sf), t=v(1-s(1-f))
_R_MAP = np.array([0, 1, 2, 2, 3, 0], dtype=np.int32)
_G_MAP = np.array([3, 0, 0, 1, 2, 2], dtype=np.int32)
_B_MAP = np.array([2, 2, 3, 0, 0, 1], dtype=np.int32)

# ---------------------------------------------------------------------------
#  Presets – each defines autonomous source behaviour
# ---------------------------------------------------------------------------
PRESETS = {
    "Cosmic Nebula": {
        "num_sources": 8,
        "speed": 0.3,
        "radius": 0.45,
        "force": 5.0,
        "density": 2.0,
        "brush_size": 45,
        "colors": [(0.9, 0.2, 0.5), (0.2, 0.5, 1.0), (0.2, 1.0, 0.7), (1.0, 0.5, 0.2)],
    },
    "Rainbow Chaos": {
        "num_sources": 10,
        "speed": 0.5,
        "radius": 0.5,
        "force": 6.0,
        "density": 2.5,
        "brush_size": 35,
        "colors": "rainbow",
    },
    "Toxic Swamp": {
        "num_sources": 6,
        "speed": 0.2,
        "radius": 0.35,
        "force": 3.0,
        "density": 3.0,
        "brush_size": 60,
        "colors": [(0.1, 0.8, 0.4), (0.0, 0.5, 0.2), (0.6, 0.9, 0.1), (0.2, 0.2, 0.1)],
    },
    "Fire & Ice": {
        "num_sources": 8,
        "speed": 0.6,
        "radius": 0.4,
        "force": 7.0,
        "density": 2.0,
        "brush_size": 40,
        "colors": [(1.0, 0.3, 0.0), (1.0, 0.8, 0.2), (0.2, 0.4, 1.0), (0.8, 0.8, 1.0)],
    },
}

# ---------------------------------------------------------------------------
#  Mouse Interaction Defaults
# ---------------------------------------------------------------------------
MOUSE_FORCE = 8.0          # Velocity injection strength on click/drag
MOUSE_DENSITY = 4.0        # Density injection amount
MOUSE_RADIUS = 18.0        # Brush radius in grid cells
MOUSE_COLOR = (0.4, 0.7, 1.0)  # Default injected color (light blue)

# ---------------------------------------------------------------------------
#  Video Export
# ---------------------------------------------------------------------------
EXPORT_WIDTH = 1920
EXPORT_HEIGHT = 1080
EXPORT_CRF = 20
EXPORT_PRESET = "fast"

# ---------------------------------------------------------------------------
#  Warm-up
# ---------------------------------------------------------------------------
WARMUP_STEPS = 200         # Steps to run before showing window
WARMUP_SPEED_MULT = 2.0    # Time multiplier during warmup
MINI_WARMUP_STEPS = 100    # Steps for preset-change mini-warmup