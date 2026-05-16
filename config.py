# config.py
import numpy as np

# Simulation Grid Size
SIM_GRID_SIZE = 512
FPS = 60

# Physics Constants
ITERATIONS = 12        
DISSIPATION = 0.999     
VELOCITY_DISSIPATION = 0.99
VORTICITY_STRENGTH = 35.0  # Increased significantly to keep fluid alive

# Visualization Modes
RENDER_MODES = ["Vibrant Oil", "Neon Glow", "Velocity Field"]

PRESETS = {
    "Cosmic Nebula": {
        "num_sources": 8,
        "speed": 0.3,
        "radius": 0.45,     
        "force": 5.0,
        "density": 2.0,
        "brush_size": 45,   
        "colors": [(0.9, 0.2, 0.5), (0.2, 0.5, 1.0), (0.2, 1.0, 0.7), (1.0, 0.5, 0.2)]
    },
    "Rainbow Chaos": {
        "num_sources": 10,
        "speed": 0.5,
        "radius": 0.5,
        "force": 6.0,
        "density": 2.5,
        "brush_size": 35,
        "colors": "rainbow"
    },
    "Toxic Swamp": {
        "num_sources": 6,
        "speed": 0.2,
        "radius": 0.35,
        "force": 3.0,
        "density": 3.0,
        "brush_size": 60,
        "colors": [(0.1, 0.8, 0.4), (0.0, 0.5, 0.2), (0.6, 0.9, 0.1), (0.2, 0.2, 0.1)]
    },
    "Fire & Ice": {
        "num_sources": 8,
        "speed": 0.6,
        "radius": 0.4,
        "force": 7.0,
        "density": 2.0,
        "brush_size": 40,
        "colors": [(1.0, 0.3, 0.0), (1.0, 0.8, 0.2), (0.2, 0.4, 1.0), (0.8, 0.8, 1.0)]
    }
}