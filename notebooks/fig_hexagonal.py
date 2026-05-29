import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')   # headless rendering
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle, FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import label as ndlabel

from phase_diagram import (
    topological_leverage,
    sweep_2x2_tiling,
    classify_region_numeric,
    L1_DEG,
    L2_DEG,
    DEG,
)
from pgsc import (
    build_pgsc,
    frenkel_coupling,
    w4_positions,
    w4_pgsc,
    w4_betti,
)
from spectral_bottleneck import fiedler_tsn_map, bright_state_osc_diff_map
from two_des import (
    linear_absorption,
    two_des_spectrum,
    transition_dipoles,
    solve_w4,
    w4_couplings,
    symmetric_block,
)

from fig_paper1 import Wheel

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

K_s = 1.0

class HexagonalWheelTiling:
    def __init__(self, rows, cols, separation=3.0, scale=1.0, angles=None):
        self.rows = rows
        self.cols = cols
        self.separation = separation
        self.scale = scale
        self.wheels = []
        for i in range(rows):
            row = []
            for j in range(cols):
                wheel = Wheel(
                    center=(
                        separation * (1.5 * j + ((i + j) % 2) * 0.5),
                        separation * (np.sqrt(3) * i / 2),
                        ),
                    scale=self.scale, n=3, angles=angles, offset = ((i+j+1)%2)*np.pi/3
                )
                row.append(wheel)
            self.wheels.append(row)
        # Create edges between adjacent wheels
        # First: the horizontal edges
        btwn_wheel_connections = []
        for i in range(rows):
            for j in range(cols // 2):
                if (i%2) == 0:
                    if 2 * j + 2 > cols - 1:
                        continue
                    btwn_wheel_connections.append((
                        (i, 2 * j + 1, 1),  # wheel on the left
                        (i, 2 * j + 2, 2)   # wheel on the right
                    ))
                if (i%2) == 1:
                    if 2 * j + 1 > cols - 1:
                        continue
                    btwn_wheel_connections.append((
                        (i, 2 * j, 1),      # wheel on the left
                        (i, 2 * j + 1, 2)   # wheel on the right
                    ))
        #Next: the vertical edges
        for i in range(rows - 1):
            for j in range(cols):
                if (i+j)%2 == 0:
                    btwn_wheel_connections.append((
                        (i, j, 1),          # wheel on the top
                        (i + 1, j, 3)       # wheel on the bottom
                    ))
                if (i+j)%2 == 1:
                    btwn_wheel_connections.append((
                        (i, j, 2),          # wheel on the top
                        (i + 1, j, 3)   # wheel on the bottom
                    ))
        self.btwn_wheel_connections = btwn_wheel_connections
    
    def calculate_active_edges(self):
        for row in self.wheels:
            for wheel in row:
                wheel.calc_active_edges()
        self.active_btwn_wheel_edges = []
        for (i1, j1, s1), (i2, j2, s2) in self.btwn_wheel_connections:
            wheel1 = self.wheels[i1][j1]
            wheel2 = self.wheels[i2][j2]
            phi1 = wheel1.spoke_angles[s1 - 1]
            phi2 = wheel2.spoke_angles[s2 - 1]
            theta_ij = np.arctan2(wheel2.spokes[s2 - 1][1] - wheel1.spokes[s1 - 1][1], wheel2.spokes[s2 - 1][0] - wheel1.spokes[s1 - 1][0])
            J = frenkel_coupling(phi1, phi2, theta_ij, 1)
            if J < 0:
                self.active_btwn_wheel_edges.append(((i1, j1, s1), (i2, j2, s2)))
    
    def render_grid(self, ax, draw_dipoles=False):
        for i in self.wheels:
            for wheel in i:
                wheel.render_wheel(ax, draw_dipoles=draw_dipoles)
        for (i1, j1, s1), (i2, j2, s2) in self.btwn_wheel_connections:
            x1, y1 = self.wheels[i1][j1].spokes[s1 - 1]
            x2, y2 = self.wheels[i2][j2].spokes[s2 - 1]
            if ((i1, j1, s1), (i2, j2, s2)) in self.active_btwn_wheel_edges:
                ax.plot([x1, x2], [y1, y2], color='#1a9850', linewidth=5)
            else:
                ax.plot([x1, x2], [y1, y2], color='#EEEEEE', linewidth=3)



if __name__ == "__main__":
    tiling = HexagonalWheelTiling(rows=12, cols=6, separation=3.0, scale=1.0, angles = [60 * DEG] + [0] * 3)
    tiling.calculate_active_edges()
    fig, ax = plt.subplots(figsize=(10, 10))
    tiling.render_grid(ax, draw_dipoles=True)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hexagonal_wheel_tiling.png'), dpi=300)





