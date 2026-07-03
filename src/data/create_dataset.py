"""
Builds Bird_Drone.xlsx from signal_generator.py.
Generates many independent 1-second signal segments per class,
concatenates them, and saves to Excel with proper label/value columns.

Run this from the PROJECT ROOT:
    python src\\data\\create_dataset.py
"""

import numpy as np
import pandas as pd
import os
from signal_generator import generate_bird_signal, generate_drone_signal

FS = 1000
DURATION = 1.0
N_SEGMENTS_PER_CLASS = 20   # each segment = 1000 samples -> 20*1000 = 20000 samples per class

np.random.seed(42)

bird_segments = [generate_bird_signal(duration=DURATION, fs=FS) for _ in range(N_SEGMENTS_PER_CLASS)]
drone_segments = [generate_drone_signal(duration=DURATION, fs=FS) for _ in range(N_SEGMENTS_PER_CLASS)]

bird_values = np.concatenate(bird_segments)
drone_values = np.concatenate(drone_segments)

print("Bird values  -> std:", bird_values.std(), "min/max:", bird_values.min(), bird_values.max())
print("Drone values -> std:", drone_values.std(), "min/max:", drone_values.min(), drone_values.max())

rows = []
for v in bird_values:
    rows.append({"label": 0, "value": v})
for v in drone_values:
    rows.append({"label": 1, "value": v})

df = pd.DataFrame(rows)

# SCRIPT_DIR = .../src/data  -> go up TWO levels to reach the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
out_path = os.path.join(PROJECT_ROOT, "data", "raw", "Bird_Drone.xlsx")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_excel(out_path, index=False)

print(f"\nSaved {len(df)} rows to {out_path}")
print(df["label"].value_counts())