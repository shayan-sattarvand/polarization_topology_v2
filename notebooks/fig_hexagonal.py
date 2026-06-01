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
from cc_density import DisjointSetUnion, print_bin

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'hexagonal')
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
        self.active_btwn_wheel_edges = []
        self.horiz_active = False
        for i in range(rows):
            for j in range(cols // 2):
                if (i%2) == 0:
                    if 2 * j + 2 > cols - 1:
                        continue
                    wheel1 = self.wheels[i][2 * j + 1]
                    wheel2 = self.wheels[i][2 * j + 2]
                    phi1 = wheel1.spoke_angles[0]
                    phi2 = wheel2.spoke_angles[1]
                    theta_ij = np.arctan2(wheel2.spokes[1][1] - wheel1.spokes[0][1], wheel2.spokes[1][0] - wheel1.spokes[0][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                    if J < 0:
                        self.active_btwn_wheel_edges.append(((i, 2 * j + 1, 1), (i, 2 * j + 2, 2)))
                        self.horiz_active = True
                    btwn_wheel_connections.append((
                        (i, 2 * j + 1, 1),  # wheel on the left
                        (i, 2 * j + 2, 2),   # wheel on the right
                        J
                    ))
                if (i%2) == 1:
                    if 2 * j + 1 > cols - 1:
                        continue
                    wheel1 = self.wheels[i][2 * j]
                    wheel2 = self.wheels[i][2 * j + 1]
                    phi1 = wheel1.spoke_angles[0]
                    phi2 = wheel2.spoke_angles[1]
                    theta_ij = np.arctan2(wheel2.spokes[1][1] - wheel1.spokes[0][1], wheel2.spokes[1][0] - wheel1.spokes[0][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                    if J < 0:
                        self.active_btwn_wheel_edges.append(((i, 2 * j, 1), (i, 2 * j + 1, 2)))
                        self.horiz_active = True
                    btwn_wheel_connections.append((
                        (i, 2 * j, 1),      # wheel on the left
                        (i, 2 * j + 1, 2),   # wheel on the right
                        J
                    ))
        #Next: the vertical edges
        self.diag1_active = False
        self.diag2_active = False
        for i in range(rows - 1):
            for j in range(cols):
                if (i+j)%2 == 0:
                    wheel1 = self.wheels[i][j]
                    wheel2 = self.wheels[i + 1][j]
                    phi1 = wheel1.spoke_angles[0]
                    phi2 = wheel2.spoke_angles[2]
                    theta_ij = np.arctan2(wheel2.spokes[2][1] - wheel1.spokes[0][1], wheel2.spokes[2][0] - wheel1.spokes[0][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                    if J < 0:
                        self.active_btwn_wheel_edges.append(((i, j, 1), (i + 1, j, 3)))
                        self.diag1_active = True
                    btwn_wheel_connections.append((
                        (i, j, 1),          # wheel on the bottom
                        (i + 1, j, 3),       # wheel on the top
                        J
                    ))
                if (i+j)%2 == 1:
                    btwn_wheel_connections.append((
                        (i, j, 2),          # wheel on the bottom
                        (i + 1, j, 3),   # wheel on the top
                        J
                    ))
                    wheel1 = self.wheels[i][j]
                    wheel2 = self.wheels[i + 1][j]
                    phi1 = wheel1.spoke_angles[1]
                    phi2 = wheel2.spoke_angles[2]
                    theta_ij = np.arctan2(wheel2.spokes[2][1] - wheel1.spokes[1][1], wheel2.spokes[2][0] - wheel1.spokes[1][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                    if J < 0:
                        self.active_btwn_wheel_edges.append(((i, j, 2), (i + 1, j, 3)))
                        self.diag2_active = True
        self.btwn_wheel_connections = btwn_wheel_connections
    
    def render_grid(self, ax, draw_dipoles=False):
        for i in self.wheels:
            for wheel in i:
                wheel.render_wheel(ax, draw_dipoles=draw_dipoles)
        for (i1, j1, s1), (i2, j2, s2), _ in self.btwn_wheel_connections:
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
            visited.add(node)

            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        return components
    
    def beta1(self):
        n_local = self.wheels[0][0].n + 1

        def global_id(r, c, local_id):
            return (r * self.cols + c) * n_local + local_id

        active_nodes = set()
        adjacency = {}

        for r in range(self.rows):
            for c in range(self.cols):
                wheel = self.wheels[r][c]
                for u, v in wheel.active_edges:
                    gu = global_id(r, c, u)
                    gv = global_id(r, c, v)
                    active_nodes.add(gu)
                    active_nodes.add(gv)
                    adjacency.setdefault(gu, set()).add(gv)
                    adjacency.setdefault(gv, set()).add(gu)

        for (row1, col1, spoke1), (row2, col2, spoke2) in self.active_btwn_wheel_edges:
            gu = global_id(row1, col1, spoke1)
            gv = global_id(row2, col2, spoke2)
            active_nodes.add(gu)
            active_nodes.add(gv)
            adjacency.setdefault(gu, set()).add(gv)
            adjacency.setdefault(gv, set()).add(gu)

        if not active_nodes:
            return 0

        visited = set()
        components = 0

        for node in active_nodes:
            if node in visited:
                continue
            components += 1
            stack = [node]
            visited.add(node)

            while stack:
                current = stack.pop()
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        active_edges = sum(len(wheel.active_edges) for row in self.wheels for wheel in row) + len(self.active_btwn_wheel_edges)
        return active_edges - len(active_nodes) + components

    def binary_representation(self):
        binary_representation = 0
        wheel1 = self.wheels[1][0]
        wheel2 = self.wheels[1][1]
        bin1 = wheel1.binary_representation()
        bin2 = wheel2.binary_representation()
        for i in f'{bin1:06b}':
            binary_representation = (binary_representation << 1) | int(i)
        for i in f'{bin2:06b}':
            binary_representation = (binary_representation << 1) | int(i)
        for i in [self.horiz_active, self.diag1_active, self.diag2_active]:
            binary_representation = (binary_representation << 1) | int(i)
        return binary_representation

    def cc_density(self):
        dsu = DisjointSetUnion(8)
        binary_representation = self.binary_representation()
        inter_wheel = binary_representation % 8
        binary_representation = binary_representation // 8
        wheel2_rep = binary_representation % 64
        binary_representation = binary_representation // 64
        wheel1_rep = binary_representation % 64
        binary_representation = binary_representation // 64
        
        for i in range(3): #model the hex lattice as a rhombic lattice with 2 horizontally adjacent wheels = 1 vertex
            if wheel1_rep & (1 << (5 - i)):
                dsu.union(0, i + 1)
        for i in range(3):
            if wheel1_rep & (1 << (2 - i)):
                dsu.union(i + 1, (i + 1) % 3 + 1)
        for i in range(3):
            if wheel2_rep & (1 << (5 - i)):
                dsu.union(4, i + 5)
        for i in range(3):
            if wheel2_rep & (1 << (2 - i)):
                dsu.union(i + 5, (i + 1) % 3 + 5)
        if inter_wheel & 4:
            dsu.union(1, 6)
        inter_wheel_new = inter_wheel
        if inter_wheel & 4 and ((inter_wheel & 2 and dsu.query(3, 5)) or (inter_wheel & 1 and dsu.query(2, 7))): #all the diag ones
            path_loc = dsu.find(1) #must go through this choke point
            if inter_wheel & 2 and dsu.query(3, 5):
                inter_wheel_new -= 2
            if inter_wheel & 1 and dsu.query(2, 7):
                inter_wheel_new -= 1
            for i in range(3):
                if dsu.find(i + 1) == path_loc and dsu.find(0) == path_loc: #center-spoke connections of wheel 1
                    wheel1_rep -= (1 << (5 - i))
                if dsu.find(i + 5) == path_loc and dsu.find(4) == path_loc: #center-spoke connections of wheel 2
                    wheel2_rep -= (1 << (5 - i))
                if dsu.find(i + 1) == path_loc and dsu.find((i + 1) % 3 + 1) == path_loc: #rim wheel 1
                    wheel1_rep -= (1 << (2 - i))
                if dsu.find(i + 5) == path_loc and dsu.find((i + 1) % 3 + 5) == path_loc: #rim wheel 2
                    wheel2_rep -= (1 << (2 - i))
        #now check vertical ones
        if inter_wheel & 3 == 3:
            #vertical column 0
            inter_wheel_new -= 3
            if dsu.query(2, 3):
                path_loc = dsu.find(2)
                for i in range(3): #don't double subtract! add the extra "and" condition for this reason
                    if dsu.find(i + 1) == path_loc and dsu.find(0) == path_loc and wheel1_rep & (1 << (5 - i)): #center-spoke connections of wheel 1
                        wheel1_rep -= (1 << (5 - i))
                    if dsu.find(i + 5) == path_loc and dsu.find(4) == path_loc and wheel2_rep & (1 << (5 - i)): #center-spoke connections of wheel 2
                        wheel2_rep -= (1 << (5 - i))
                    if dsu.find(i + 1) == path_loc and dsu.find((i + 1) % 3 + 1) == path_loc and wheel1_rep & (1 << (2 - i)): #rim wheel 1
                        wheel1_rep -= (1 << (2 - i))
                    if dsu.find(i + 5) == path_loc and dsu.find((i + 1) % 3 + 5) == path_loc and wheel2_rep & (1 << (2 - i)): #rim wheel 2
                        wheel2_rep -= (1 << (2 - i))
            if dsu.query(5, 7):
                path_loc = dsu.find(5)
                for i in range(3): #don't double subtract! add the extra "and" condition for this reason
                    if dsu.find(i + 1) == path_loc and dsu.find(0) == path_loc and wheel1_rep & (1 << (5 - i)): #center-spoke connections of wheel 1
                        wheel1_rep -= (1 << (5 - i))
                    if dsu.find(i + 5) == path_loc and dsu.find(4) == path_loc and wheel2_rep & (1 << (5 - i)): #center-spoke connections of wheel 2
                        wheel2_rep -= (1 << (5 - i))
                    if dsu.find(i + 1) == path_loc and dsu.find((i + 1) % 3 + 1) == path_loc and wheel1_rep & (1 << (2 - i)): #rim wheel 1
                        wheel1_rep -= (1 << (2 - i))
                    if dsu.find(i + 5) == path_loc and dsu.find((i + 1) % 3 + 5) == path_loc and wheel2_rep & (1 << (2 - i)): #rim wheel 2
                        wheel2_rep -= (1 << (2 - i))
        dsu_now = DisjointSetUnion(8)
        for i in range(3): #model the hex lattice as a rhombic lattice with 2 horizontally adjacent wheels = 1 vertex
            if wheel1_rep & (1 << (5 - i)):
                dsu_now.union(0, i + 1)
        for i in range(3):
            if wheel1_rep & (1 << (2 - i)):
                dsu_now.union(i + 1, (i + 1) % 3 + 1)
        for i in range(3):
            if wheel2_rep & (1 << (5 - i)):
                dsu_now.union(4, i + 5)
        for i in range(3):
            if wheel2_rep & (1 << (2 - i)):
                dsu_now.union(i + 5, (i + 1) % 3 + 5)
        if inter_wheel_new & 4:
            dsu_now.union(1, 6)
        if inter_wheel_new & 2:
            dsu_now.union(3, 5)
        if inter_wheel_new & 1:
            dsu_now.union(2, 7)
        return dsu_now.connected_components_edges() / 2 # divide by 2 to account for the fact that we modeled the hex lattice as a rhombic lattice with 2 horizontally adjacent wheels = 1 vertex
    
    def beta1_density(self):
        e = sum(1 for i in str(self.binary_representation) if i == '1') / 2
        v = 4
        c = self.cc_density()
        return (e - v + c)

def fig1(save=True):
    '''Generate a figure showing three wheels, all with n=3 spokes and spoke angles 60 degrees, but with different center angles.'''
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    theta_C_values = [0, 90 * DEG, 180 * DEG]
    for ax, theta_C in zip(axes, theta_C_values):
        wheel = Wheel(center=(0, 0), scale=1.0, n=3, angles=[theta_C] + [60 * DEG] * 3)
        wheel.render_wheel(ax, draw_dipoles=True)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"$\\theta_C = {int(theta_C / DEG)}^\\circ$")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p3_1_concept.png"), dpi=300)




def fig2_mechanism_hexagonal(save=True):
    '''Generate a figure showing 3 different 5x5 hexagonal grids with n=6 spokes and spoke angles 60 degrees, but with different center angles.'''
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    wheel_grid1 = HexagonalWheelTiling(rows=5, cols=5, separation=3.0, scale=1.0, angles=[DEG * 0] + [60 * DEG] * 3)
    wheel_grid2 = HexagonalWheelTiling(rows=5, cols=5, separation=3.0, scale=1.0, angles=[DEG * 90] + [60 * DEG] * 3)
    wheel_grid3 = HexagonalWheelTiling(rows=5, cols=5, separation=3.0, scale=1.0, angles=[DEG * 180] + [60 * DEG] * 3)

    for ax in axes[1]:
        ax.set_aspect('equal')
        ax.axis('off')

    wheel_grid1.render_grid(axes[1][0], draw_dipoles=True)
    wheel_grid2.render_grid(axes[1][1], draw_dipoles=True)
    wheel_grid3.render_grid(axes[1][2], draw_dipoles=True)
    axes[1][0].set_title(f"Grid with $\\theta_C = 0$° \n$\\beta_1 = {wheel_grid1.beta1()}$")
    axes[1][1].set_title(f"Grid with $\\theta_C = 90$° \n$\\beta_1 = {wheel_grid2.beta1()}$")
    axes[1][2].set_title(f"Grid with $\\theta_C = 180$° \n$\\beta_1 = {wheel_grid3.beta1()}$")

    theta_c_values = np.linspace(0, 180, 200) * DEG
    J_values_axis1 = [frenkel_coupling(theta_c, 60 * DEG, 0, 1) for theta_c in theta_c_values]
    J_values_axis2 = [frenkel_coupling(theta_c, 60 * DEG, 2 * np.pi / 3, 1) for theta_c in theta_c_values]
    J_values_axis3 = [frenkel_coupling(theta_c, 60 * DEG, 4 * np.pi / 3, 1) for theta_c in theta_c_values]
    axes[0][0].plot(theta_c_values / DEG, J_values_axis1, color='blue')
    axes[0][0].plot(theta_c_values / DEG, J_values_axis2, color='orange')
    axes[0][0].plot(theta_c_values / DEG, J_values_axis3, color='green')
    axes[0][0].axhline(0, color='gray', linestyle='--')
    axes[0][0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][0].set_ylabel("$J / K_s$")
    axes[0][0].set_title("Coupling strength vs. center angle for spoke axes")
    axes[0][0].legend(["Axis 1", "Axis 2", "Axis 3"])

    center_angles = np.linspace(0, 180, 200) * DEG
    beta0_values = []
    beta1_values = []
    for theta_c in center_angles:
        grid = HexagonalWheelTiling(rows=5, cols=5, separation=3.0, scale=1.0, angles=[theta_c] + [60 * DEG] * 3)
        beta0_values.append(grid.beta0())
        beta1_values.append(grid.beta1())

    axes[0][1].plot(center_angles / DEG, beta1_values, color='purple')
    axes[0][1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][1].set_ylabel("Independent loops $\\beta_1$")
    axes[0][1].set_title("Grid loop topology vs. center angle")
    axes[0][1].set_ylim(0, max(beta1_values) + 1)

    axes[0][2].plot(center_angles / DEG, beta0_values, color='green')
    axes[0][2].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][2].set_ylabel("Connected components $\\beta_0$")
    axes[0][2].set_title("Grid connectivity vs. center angle")

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p3_2_mechanism.png"), dpi=300)

def fig_2_phase_diagram_hexagonal(save=True):
    '''Phase diagram of $\\beta_0$ and $\\beta_1$ for a single wheel (n=3) vs center and spoke angles.'''
    theta_c_values = np.linspace(0, 360, 200) * DEG
    spoke_angles = np.linspace(0, 360, 200) * DEG
    Theta_C, Spoke_Angle = np.meshgrid(theta_c_values, spoke_angles)
    beta0_values = np.zeros_like(Theta_C, dtype=int)
    beta1_values = np.zeros_like(Theta_C, dtype=int)

    for i in range(Theta_C.shape[0]):
        for j in range(Theta_C.shape[1]):
            wheel = Wheel(center=(0, 0), scale=1.0, n=3, angles=[Theta_C[i, j]] + [Spoke_Angle[i, j]] * 3)
            beta0_values[i, j] = wheel.beta0()
            beta1_values[i, j] = wheel.beta1()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c0 = axes[0].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta0_values, shading='auto', cmap='viridis')
    fig.colorbar(c0, ax=axes[0], label='$\\beta_0$')
    axes[0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0].set_ylabel("Spoke angle (degrees)")
    axes[0].set_title("Phase diagram of $\\beta_0$ for a single 3-wheel")

    c1 = axes[1].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta1_values, shading='auto', cmap='viridis')
    fig.colorbar(c1, ax=axes[1], label='$\\beta_1$')
    axes[1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[1].set_ylabel("Spoke angle (degrees)")
    axes[1].set_title("Phase diagram of $\\beta_1$ for a single 3-wheel")

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p3_2_phase_diagram.png"), dpi=300)

def fig_2_5_phase_diagram_hexagonal(save=True):
    '''Phase diagram of $\\beta_0$ and $\\beta_1$ for a 5x5 hexagonal grid vs center and spoke angles.'''
    theta_c_values = np.linspace(0, 360, 200) * DEG
    spoke_angles = np.linspace(0, 360, 200) * DEG
    Theta_C, Spoke_Angle = np.meshgrid(theta_c_values, spoke_angles)
    beta0_values = np.zeros_like(Theta_C, dtype=int)
    beta1_values = np.zeros_like(Theta_C, dtype=int)

    for i in range(Theta_C.shape[0]):
        for j in range(Theta_C.shape[1]):
            grid = HexagonalWheelTiling(rows=5, cols=5, separation=3.0, scale=1.0, angles=[Theta_C[i, j]] + [Spoke_Angle[i, j]] * 3)
            beta0_values[i, j] = grid.beta0()
            beta1_values[i, j] = grid.beta1()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c0 = axes[0].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta0_values, shading='auto', cmap='viridis')
    fig.colorbar(c0, ax=axes[0], label='$\\beta_0$')
    axes[0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0].set_ylabel("Spoke angle (degrees)")
    axes[0].set_title("Phase diagram of $\\beta_0$ for a 5x5 hexagonal grid")

    c1 = axes[1].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta1_values, shading='auto', cmap='viridis')
    fig.colorbar(c1, ax=axes[1], label='$\\beta_1$')
    axes[1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[1].set_ylabel("Spoke angle (degrees)")
    axes[1].set_title("Phase diagram of $\\beta_1$ for a 5x5 hexagonal grid")

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p3_2_5_phase_diagram.png"), dpi=300)


if __name__ == "__main__":
    # print("Generating figure 1...")
    # fig1(save=True)
    # print("Generating figure 2 mechanism...")
    # fig2_mechanism_hexagonal(save=True)
    # print("Generating figure 2 phase diagram...")
    # fig_2_phase_diagram_hexagonal(save=True)
    # print("Generating figure 2.5 phase diagram...")
    # fig_2_5_phase_diagram_hexagonal(save=True)
    test_grid = HexagonalWheelTiling(rows=5, cols=5, separation=3.0, scale=1.0, angles=[DEG * 180] + [60 * DEG] * 2 + [120 * DEG])
    fig, ax = plt.subplots(figsize=(6, 6))
    test_grid.render_grid(ax, draw_dipoles=True)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(os.path.join(FIGURES_DIR, "test_grid.png"), dpi=300)
    print(test_grid.cc_density())





