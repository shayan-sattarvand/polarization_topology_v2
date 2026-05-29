import sys
import os
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

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

from fig_square import Wheel

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'kagome')
os.makedirs(FIGURES_DIR, exist_ok=True)

K_s = 1.0

from math import sqrt

def kagome_map(x, y, scale):
    s = y % 3
    n = (y - s) // 3

    # horizontal row shear, but wrapped every 2 triangular rows
    p = n % 2

    bx = (7*s - 3*s*s) / 4
    by = sqrt(3) * s * (s - 1) / 4

    X = 2*x + p + bx
    Y = sqrt(3)*n + by
    X *= scale
    Y *= scale 

    return X, Y

class KagomeWheelTiling:
    def __init__(self, rows, cols, separation=3.0, scale=1.0, angles=None):
        if rows * cols == 0:
            raise ValueError("Number of rows and columns must be greater than 0.")
        self.rows = rows
        self.cols = cols
        self.separation = separation
        self.scale = scale
        self.wheels = []
        for i in range(rows):
            row = []
            for j in range(cols):
                wheel = Wheel(
                    center=kagome_map(j, i, self.separation),
                    scale=self.scale, n=6, angles=angles,
                )
                row.append(wheel)
            self.wheels.append(row)
        # Create edges between adjacent wheels
        btwn_wheel_connections = []
        self.active_btwn_wheel_edges = []
        for i in range((rows + 2) // 3): #Row 0, 3, 6, ... are the same, so we only need to loop through one of them and then repeat the pattern every 6 rows
            row = 3 * i
            for j in range(cols):
                # Connection 1: horiz right
                if j != cols - 1 and row != rows - 1:
                    wheel1 = self.wheels[row][j]
                    wheel2 = self.wheels[row + 1][j] # Seems counterintuitive, but trust me bro
                    phi1 = wheel1.spoke_angles[0]
                    phi2 = wheel2.spoke_angles[3]
                    theta_ij = np.arctan2(wheel2.spokes[3][1] - wheel1.spokes[0][1], wheel2.spokes[3][0] - wheel1.spokes[0][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                    if J < 0:
                        self.active_btwn_wheel_edges.append(((row, j, 1), (row + 1, j, 4)))
                    btwn_wheel_connections.append(((row, j, 1), (row + 1, j, 4), J))  # connect spoke 1 of wheel (row, j) to spoke 4 of wheel (row + 1, j)
                # Connection 2: diag right-up
                if row < rows - 2:
                    wheel1 = self.wheels[row][j]
                    wheel2 = self.wheels[row + 2][j] # Seems counterintuitive, but trust me bro
                    phi1 = wheel1.spoke_angles[1]
                    phi2 = wheel2.spoke_angles[4]
                    theta_ij = np.arctan2(wheel2.spokes[4][1] - wheel1.spokes[1][1], wheel2.spokes[4][0] - wheel1.spokes[1][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                    if J < 0:
                        self.active_btwn_wheel_edges.append(((row, j, 2), (row + 2, j, 5)))
                    btwn_wheel_connections.append(((row, j, 2), (row + 2, j, 5), J))  # connect spoke 2 of wheel (row, j) to spoke 5 of wheel (row + 2, j)
        if rows > 1:
            for i in range((rows + 1) // 3): #Rows 1, 4, 7... 
                row = 3 * i + 1
                for j in range(cols):
                    # Connection 1: horiz right
                    if j != cols - 1:
                        wheel1 = self.wheels[row][j]
                        wheel2 = self.wheels[row - 1][j + 1] # Seems counterintuitive, but trust me bro
                        phi1 = wheel1.spoke_angles[0]
                        phi2 = wheel2.spoke_angles[3]
                        theta_ij = np.arctan2(wheel2.spokes[3][1] - wheel1.spokes[0][1], wheel2.spokes[3][0] - wheel1.spokes[0][0])
                        J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                        if J < 0:
                            self.active_btwn_wheel_edges.append(((row, j, 1), (row - 1, j + 1, 4)))
                        btwn_wheel_connections.append(((row, j, 1), (row - 1, j + 1, 4), J))  # connect spoke 1 of wheel (row, j) to spoke 4 of wheel (row - 1, j + 1)
                    # Connection 2: diag left-up
                    if row != rows - 1:
                        wheel1 = self.wheels[row][j]
                        wheel2 = self.wheels[row + 1][j] # Seems counterintuitive, but trust me bro
                        phi1 = wheel1.spoke_angles[2]
                        phi2 = wheel2.spoke_angles[5]
                        theta_ij = np.arctan2(wheel2.spokes[5][1] - wheel1.spokes[2][1], wheel2.spokes[5][0] - wheel1.spokes[2][0])
                        J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                        if J < 0:
                            self.active_btwn_wheel_edges.append(((row, j, 3), (row + 1, j, 6)))
                        btwn_wheel_connections.append(((row, j, 3), (row + 1, j, 6), J))  # connect spoke 3 of wheel (row, j) to spoke 6 of wheel (row + 1, j)
        if rows > 2:
            for i in range(rows // 3): #Rows 2, 5, 8... 
                row = 3 * i + 2
                for j in range(cols):
                    #Connection 1: diag right-up
                    if (row % 6) == 2:
                        if row != rows - 1:
                            wheel1 = self.wheels[row][j]
                            wheel2 = self.wheels[row + 1][j] # Seems counterintuitive, but trust me bro
                            phi1 = wheel1.spoke_angles[1]
                            phi2 = wheel2.spoke_angles[4]
                            theta_ij = np.arctan2(wheel2.spokes[4][1] - wheel1.spokes[1][1], wheel2.spokes[4][0] - wheel1.spokes[1][0])
                            J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                            if J < 0:
                                self.active_btwn_wheel_edges.append(((row, j, 2), (row + 1, j, 5)))
                            btwn_wheel_connections.append(((row, j, 2), (row + 1, j, 5), J))  # connect spoke 2 of wheel (row, j) to spoke 5 of wheel (row + 1, j)
                        #Connection 2: diag left-up
                        if row < rows - 2 and j != 0:
                            wheel1 = self.wheels[row][j]
                            wheel2 = self.wheels[row + 2][j - 1] # Seems counterintuitive, but trust me bro
                            phi1 = wheel1.spoke_angles[2]
                            phi2 = wheel2.spoke_angles[5]
                            theta_ij = np.arctan2(wheel2.spokes[5][1] - wheel1.spokes[2][1], wheel2.spokes[5][0] - wheel1.spokes[2][0])
                            J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                            if J < 0:
                                self.active_btwn_wheel_edges.append(((row, j, 3), (row + 2, j - 1, 6)))
                            btwn_wheel_connections.append(((row, j, 3), (row + 2, j - 1, 6), J))  # connect spoke 3 of wheel (row, j) to spoke 6 of wheel (row + 2, j - 1)
                    else:
                        if row != rows - 1 and j < cols - 1:
                            wheel1 = self.wheels[row][j]
                            wheel2 = self.wheels[row + 1][j + 1] # Seems counterintuitive, but trust me bro
                            phi1 = wheel1.spoke_angles[1]
                            phi2 = wheel2.spoke_angles[4]
                            theta_ij = np.arctan2(wheel2.spokes[4][1] - wheel1.spokes[1][1], wheel2.spokes[4][0] - wheel1.spokes[1][0])
                            J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                            if J < 0:
                                self.active_btwn_wheel_edges.append(((row, j, 2), (row + 1, j + 1, 5)))
                            btwn_wheel_connections.append(((row, j, 2), (row + 1, j + 1, 5), J))  # connect spoke 2 of wheel (row, j) to spoke 5 of wheel (row + 1, j)
                        #Connection 2: diag left-up
                        if row < rows - 2:
                            wheel1 = self.wheels[row][j]
                            wheel2 = self.wheels[row + 2][j] # Seems counterintuitive, but trust me bro
                            phi1 = wheel1.spoke_angles[2]
                            phi2 = wheel2.spoke_angles[5]
                            theta_ij = np.arctan2(wheel2.spokes[5][1] - wheel1.spokes[2][1], wheel2.spokes[5][0] - wheel1.spokes[2][0])
                            J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                            if J < 0:
                                self.active_btwn_wheel_edges.append(((row, j, 3), (row + 2, j, 6)))
                            btwn_wheel_connections.append(((row, j, 3), (row + 2, j, 6), J))  # connect spoke 3 of wheel (row, j) to spoke 6 of wheel (row + 2, j)
                    


        self.btwn_wheel_connections = btwn_wheel_connections
    
    def render_grid(self, ax, draw_dipoles=False):
        for i in self.wheels:
            for wheel in i:
                wheel.render_wheel(ax, draw_dipoles=draw_dipoles, node_size = 50)
        for (i1, j1, s1), (i2, j2, s2) , _ in self.btwn_wheel_connections:
            x1, y1 = self.wheels[i1][j1].spokes[s1 - 1]
            x2, y2 = self.wheels[i2][j2].spokes[s2 - 1]
            if ((i1, j1, s1), (i2, j2, s2)) in self.active_btwn_wheel_edges:
                ax.plot([x1, x2], [y1, y2], color='#1a9850', linewidth=5)
            else:
                ax.plot([x1, x2], [y1, y2], color='#EEEEEE', linewidth=3)
    
    def beta0(self):
        '''Calculate the zeroth Betti number of the tiling based on active edges.'''
        n_wheels = self.rows * self.cols
        n_local = self.wheels[0][0].n + 1
        total_nodes = n_wheels * n_local

        adjacency = {node: set() for node in range(total_nodes)}

        def global_id(r, c, local_id):
            return (r * self.cols + c) * n_local + local_id
        
        def reverse_global_id(global_id):
            local_id = global_id % n_local
            wheel_id = global_id // n_local
            r = wheel_id // self.cols
            c = wheel_id % self.cols
            return r, c, local_id

        for r in range(self.rows):
            for c in range(self.cols):
                wheel = self.wheels[r][c]
                for u, v in wheel.active_edges:
                    gu = global_id(r, c, u)
                    gv = global_id(r, c, v)
                    adjacency[gu].add(gv)
                    adjacency[gv].add(gu)

        for ((row1, col1, spoke1), (row2, col2, spoke2)) in self.active_btwn_wheel_edges:
            gu = global_id(row1, col1, spoke1)
            gv = global_id(row2, col2, spoke2)
            adjacency[gu].add(gv)
            adjacency[gv].add(gu)

        visited = set()
        components = 0

        for node in range(total_nodes):
            if node in visited:
                continue
            components += 1
            stack = [node]
            stack_locs = [reverse_global_id(node)]
            visited.add(node)

            while stack:
                current = stack.pop()
                stack_locs.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
                        stack_locs.append(reverse_global_id(neighbor))

        return components
    
    def beta1(self):
        e = len(self.active_btwn_wheel_edges) + sum(len(wheel.active_edges) for row in self.wheels for wheel in row)
        c = self.beta0()
        v = self.rows * self.cols * 7
        return e - v + c

if __name__ == "__main__":
    tiling = KagomeWheelTiling(rows=6, cols=6, separation=3.0, scale=1.0, angles=[240 * DEG] + [30 * DEG] * 6)
    print("Beta0:", tiling.beta0())
    print("Beta1:", tiling.beta1())
    fig, ax = plt.subplots(figsize=(10, 10))
    tiling.render_grid(ax, draw_dipoles=True)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Kagome wheel tiling with active edges in green")
    plt.savefig(os.path.join(FIGURES_DIR, 'kagome_wheel_tiling.png'), dpi=300)








