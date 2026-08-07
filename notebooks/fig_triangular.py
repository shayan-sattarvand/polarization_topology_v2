import sys
import os
import importlib
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
if os.environ.get("MPL_HEADLESS", "1") == "1":
    matplotlib.use("Agg")   # headless rendering
else:
    # Prefer backends that are actually usable on macOS and avoid cairo-based ones.
    for backend_name in ["TkAgg", "QtAgg", "Qt5Agg", "MacOSX", "WebAgg"]:
        try:
            importlib.import_module(f"matplotlib.backends.backend_{backend_name.lower()}")
            matplotlib.use(backend_name)
            break
        except Exception:
            continue
    else:
        matplotlib.use("Agg")
    try:
        print("matplotlib backend:", matplotlib.get_backend())
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle, FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import label as ndlabel
from multiprocessing import Pool

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

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'triangular')
os.makedirs(FIGURES_DIR, exist_ok=True)
FIG4_CACHE_DIR = os.path.join(FIGURES_DIR, 'fig4_cache')
os.makedirs(FIG4_CACHE_DIR, exist_ok=True)

K_s = 1.0
NEW_J_CONDITION = True
J_SQUARED_THRESHOLD = 1


def _fig4_cache_paths():
    metadata_path = os.path.join(FIG4_CACHE_DIR, 'metadata.json')
    panel_paths = [os.path.join(FIG4_CACHE_DIR, f'panel_{index}.csv') for index in range(4)]
    return metadata_path, panel_paths


def _save_fig4_cache(metadata, panel_values):
    metadata_path, panel_paths = _fig4_cache_paths()
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    for panel_path, panel_values_for_axis in zip(panel_paths, panel_values):
        np.savetxt(panel_path, panel_values_for_axis, fmt='%d', delimiter=',')


def _load_fig4_cache():
    metadata_path, panel_paths = _fig4_cache_paths()
    with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
        metadata = json.load(metadata_file)
    panel_values = [np.loadtxt(panel_path, delimiter=',', dtype=int) for panel_path in panel_paths]
    return metadata, panel_values

class TriangleWheelTiling:
    def __init__(self, rows, cols, separation=3.0, scale=1.0, angles=None):
        self.rows = rows
        self.cols = cols
        self.separation = separation
        self.scale = scale
        self.wheels = []
        for i in range(rows):
            row = []
            for j in range(cols):
                center = ((j + 0.5 * (i % 2)) * separation, i * separation * 3 ** 0.5 / 2)  # staggered vertical spacing for triangular tiling
                wheel = Wheel(center, scale, 6, angles=angles)
                row.append(wheel)
            self.wheels.append(row)
        btwn_wheel_connections = []
        self.active_btwn_wheel_edges = []
        self.horiz_active = False
        self.diag1_active = False
        self.diag2_active = False
        # Horizontal first
        for i in range(rows):
            for j in range(cols - 1):
                wheel1 = self.wheels[i][j]
                wheel2 = self.wheels[i][j + 1]
                phi1 = wheel1.spoke_angles[0]
                phi2 = wheel2.spoke_angles[3]
                theta_ij = np.arctan2(wheel2.spokes[3][1] - wheel1.spokes[0][1], wheel2.spokes[3][0] - wheel1.spokes[0][0])
                J = frenkel_coupling(phi1, phi2, theta_ij, separation - 2 * scale)
                if NEW_J_CONDITION:
                    if J ** 2 > J_SQUARED_THRESHOLD:
                        self.active_btwn_wheel_edges.append(((i, j, 1), (i, j + 1, 4)))
                        self.horiz_active = True
                else:
                    if J < 0:
                        self.active_btwn_wheel_edges.append(((i, j, 1), (i, j + 1, 4)))
                        self.horiz_active = True
                btwn_wheel_connections.append(((i, j, 1), (i, j + 1, 4), J))  # connect spoke 1 of wheel (i, j) to spoke 4 of wheel (i, j + 1)
        # Then diagonal right-up
        for i in range(rows - 1):
            for j in range(cols):
                if i % 2 == 1 and j == cols - 1:
                    continue  # skip rightmost column for odd rows
                if i% 2 == 0:
                    wheel1 = self.wheels[i][j]
                    wheel2 = self.wheels[i + 1][j]
                    phi1 = wheel1.spoke_angles[1]
                    phi2 = wheel2.spoke_angles[4]
                    theta_ij = np.arctan2(wheel2.spokes[4][1] - wheel1.spokes[1][1], wheel2.spokes[4][0] - wheel1.spokes[1][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, separation - 2 * scale)
                    if NEW_J_CONDITION:
                        if J ** 2 > J_SQUARED_THRESHOLD:
                            self.active_btwn_wheel_edges.append(((i, j, 2), (i + 1, j, 5)))
                            self.diag1_active = True
                    else:
                        if J < 0:
                            self.active_btwn_wheel_edges.append(((i, j, 2), (i + 1, j, 5)))
                            self.diag1_active = True
                    btwn_wheel_connections.append(((i, j, 2), (i + 1, j, 5), J))  # connect spoke 2 of wheel (i, j) to spoke 5 of wheel above-right
                else:
                    wheel1 = self.wheels[i][j]
                    wheel2 = self.wheels[i + 1][j + 1]
                    phi1 = wheel1.spoke_angles[1]
                    phi2 = wheel2.spoke_angles[4]
                    theta_ij = np.arctan2(wheel2.spokes[4][1] - wheel1.spokes[1][1], wheel2.spokes[4][0] - wheel1.spokes[1][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, separation - 2 * scale)
                    if NEW_J_CONDITION:
                        if J ** 2 > J_SQUARED_THRESHOLD:
                            self.active_btwn_wheel_edges.append(((i, j, 2), (i + 1, j + 1, 5)))
                            self.diag1_active = True
                    else:
                        if J < 0:
                            self.active_btwn_wheel_edges.append(((i, j, 2), (i + 1, j + 1, 5)))
                            self.diag1_active = True
                    btwn_wheel_connections.append(((i, j, 2), (i + 1, j + 1, 5), J))  # connect spoke 2 of wheel (i, j) to spoke 5 of wheel above-right
        # Then diagonal left-up
        for i in range(rows - 1):
            for j in range(cols):
                if i % 2 == 0 and j == 0:
                    continue  # skip leftmost column for even rows
                if i% 2 == 0:
                    wheel1 = self.wheels[i][j]
                    wheel2 = self.wheels[i + 1][j - 1]
                    phi1 = wheel1.spoke_angles[2]
                    phi2 = wheel2.spoke_angles[5]
                    theta_ij = np.arctan2(wheel2.spokes[5][1] - wheel1.spokes[2][1], wheel2.spokes[5][0] - wheel1.spokes[2][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, separation - 2 * scale)
                    if NEW_J_CONDITION:
                        if J ** 2 > J_SQUARED_THRESHOLD:
                            self.active_btwn_wheel_edges.append(((i, j, 3), (i + 1, j - 1, 6)))
                            self.diag2_active = True
                    else:
                        if J < 0:
                            self.active_btwn_wheel_edges.append(((i, j, 3), (i + 1, j - 1, 6)))
                            self.diag2_active = True
                    btwn_wheel_connections.append(((i, j, 3), (i + 1, j - 1, 6), J))  # connect spoke 3 of wheel (i, j) to spoke 6 of wheel above-left
                else:
                    wheel1 = self.wheels[i][j]
                    wheel2 = self.wheels[i + 1][j]
                    phi1 = wheel1.spoke_angles[2]
                    phi2 = wheel2.spoke_angles[5]
                    theta_ij = np.arctan2(wheel2.spokes[5][1] - wheel1.spokes[2][1], wheel2.spokes[5][0] - wheel1.spokes[2][0])
                    J = frenkel_coupling(phi1, phi2, theta_ij, separation - 2 * scale)
                    if NEW_J_CONDITION:
                        if J ** 2 > J_SQUARED_THRESHOLD:
                            self.active_btwn_wheel_edges.append(((i, j, 3), (i + 1, j, 6)))
                            self.diag2_active = True
                    else:
                        if J < 0:
                            self.active_btwn_wheel_edges.append(((i, j, 3), (i + 1, j, 6)))
                            self.diag2_active = True
                    btwn_wheel_connections.append(((i, j, 3), (i + 1, j, 6), J))  # connect spoke 3 of wheel (i, j) to spoke 6 of wheel above-left
        self.btwn_wheel_connections = btwn_wheel_connections

    
    def render_grid(self, ax, draw_dipoles=False, node_size = 100):
        for i in self.wheels:
            for wheel in i:
                wheel.render_wheel(ax, draw_dipoles=draw_dipoles, node_size=node_size)
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
        '''Return a binary representation of the tiling's center wheels. First 12 digits are the wheel representation, next 3 digits are the horizontal, diagonal1, and diagonal2 between-wheel connections.'''
        wheel = self.wheels[1][1]
        num = wheel.binary_representation()
        for i in [self.horiz_active, self.diag1_active, self.diag2_active]:
            num *= 2
            if i:
                num += 1
        return num

    def num_triangles(self):
        bin_rep = f'{self.binary_representation():15b}'[:-3]  # first 12 bits are the wheel, next 3 bits are the between-wheel connections
        count = 0
        for i in range(6):
            if bin_rep[i] == '1' and bin_rep[(i + 1) % 6] == '1' and bin_rep[6 + i] == '1':
                count += 1
        return count

    def euler_characteristic_density(self):
        '''Calculate the Euler characteristic of the tiling based on active edges.'''
        e = sum(1 for i in f'{self.binary_representation():b}' if i == '1')
        v = 7
        t = self.num_triangles()
        return v - e + t
            
    def cc_density(self, include_lone=True):
        '''Connected component density of the lattice'''
        binary_representation = self.binary_representation()
        dsu = DisjointSetUnion(7)
        inter_wheel = binary_representation % 8
        wheel_rep = binary_representation // 8
        binary_representation = binary_representation // 8
        rim_connections = binary_representation % 64
        binary_representation = binary_representation // 64
        wheel_connections = binary_representation
        for i in range(6):
            if rim_connections & (1 << (5 - i)):
                dsu.union(i + 1, (i + 1) % 6 + 1)
        for i in range(6):
            if wheel_connections & (1 << (5 - i)):
                dsu.union(0, i + 1)
        inter_wheel_new = inter_wheel
        if (inter_wheel & 4 and dsu.query(1, 4)) or (inter_wheel & 2 and dsu.query(2,5)) or (inter_wheel & 1 and dsu.query(3,6)):
            if inter_wheel & 4 and dsu.query(1, 4):
                inter_wheel_new -= 4
            if inter_wheel & 2 and dsu.query(2,5):
                inter_wheel_new -= 2
            if inter_wheel & 1 and dsu.query(3,6):
                inter_wheel_new -= 1
            center_loc = dsu.find(0)  # gonna be same as the path location cuz it has to go thru center
            for i in range(6):
                loc = dsu.find(i + 1)
                if loc == center_loc and wheel_rep & (1 << (11 - i)):
                    wheel_rep -= 1 << (11 - i)
                loc2 = dsu.find((i + 1) % 6 + 1)
                if loc2 == center_loc and loc == center_loc and wheel_rep & (1 << (5 - i)):
                    wheel_rep -= 1 << (5 - i)
        #ugh now time to check zig-zags
        if (inter_wheel & 5) == 5 and dsu.query(1, 3) and dsu.query(4, 6): # zigzag consisting of horiz and diag2
            inter_wheel_new -= inter_wheel_new & 5
            path_loc = dsu.find(1)
            for i in range(6):
                loc = dsu.find(i + 1)
                if loc == path_loc and wheel_rep & (1 << (11 - i)):
                    wheel_rep -= 1 << (11 - i)
                loc2 = dsu.find((i + 1) % 6 + 1)
                if loc2 == path_loc and loc == path_loc and wheel_rep & (1 << (5 - i)):
                    wheel_rep -= 1 << (5 - i)
            path_loc = dsu.find(4)
            for i in range(6):
                loc = dsu.find(i + 1)
                if loc == path_loc and wheel_rep & (1 << (11 - i)):
                    wheel_rep -= 1 << (11 - i)
                loc2 = dsu.find((i + 1) % 6 + 1)
                if loc2 == path_loc and loc == path_loc and wheel_rep & (1 << (5 - i)):
                    wheel_rep -= 1 << (5 - i)
        if (inter_wheel & 3) == 3 and dsu.query(2, 6) and dsu.query(3, 5): # zigzag consisting of diag1 and diag2
            inter_wheel_new -= inter_wheel_new & 3
            path_loc = dsu.find(2) 
            for i in range(6):
                loc = dsu.find(i + 1)
                if loc == path_loc and wheel_rep & (1 << (11 - i)):
                    wheel_rep -= 1 << (11 - i)
                loc2 = dsu.find((i + 1) % 6 + 1)
                if loc2 == path_loc and loc == path_loc and wheel_rep & (1 << (5 - i)):
                    wheel_rep -= 1 << (5 - i)
            path_loc = dsu.find(3)
            for i in range(6):
                loc = dsu.find(i + 1)
                if loc == path_loc and wheel_rep & (1 << (11 - i)):
                    wheel_rep -= 1 << (11 - i)
                loc2 = dsu.find((i + 1) % 6 + 1)
                if loc2 == path_loc and loc == path_loc and wheel_rep & (1 << (5 - i)):
                    wheel_rep -= 1 << (5 - i)
        if (inter_wheel & 6) == 6 and dsu.query(2, 4) and dsu.query(1, 5): # zigzag consisting of horiz and diag1
            inter_wheel_new -= inter_wheel_new & 6
            path_loc = dsu.find(1) 
            for i in range(6):
                loc = dsu.find(i + 1)
                if loc == path_loc and wheel_rep & (1 << (11 - i)):
                    wheel_rep -= 1 << (11 - i)
                loc2 = dsu.find((i + 1) % 6 + 1)
                if loc2 == path_loc and loc == path_loc and wheel_rep & (1 << (5 - i)):
                    wheel_rep -= 1 << (5 - i)
            path_loc = dsu.find(4)
            for i in range(6):
                loc = dsu.find(i + 1)
                if loc == path_loc and wheel_rep & (1 << (11 - i)):
                    wheel_rep -= 1 << (11 - i)
                loc2 = dsu.find((i + 1) % 6 + 1)
                if loc2 == path_loc and loc == path_loc and wheel_rep & (1 << (5 - i)):
                    wheel_rep -= 1 << (5 - i)

            
        dsu_now = DisjointSetUnion(7)
        for i in range(6):
            if wheel_rep & (1 << (5 - i)):
                dsu_now.union(i + 1, (i + 1) % 6 + 1)
        for i in range(6):
            if wheel_rep & (1 << (11 - i)):
                dsu_now.union(0, i + 1)
        for i in range(3):
            if inter_wheel_new & (1 << (2 - i)):
                dsu_now.union(i + 1, i + 4)
        if include_lone:
            return dsu_now.connected_components_edges() + 7 - self.connected_vertices()
        else:
            return dsu_now.connected_components_edges()

    def beta1_density(self):
        e = sum(1 for i in f'{self.binary_representation():b}' if i == '1')
        v = 7
        c = self.cc_density(include_lone=False)
        t = self.num_triangles()
        return max(0, e - v + c - t)
    
    def num_triangles(self):
        bin_rep = f'{self.binary_representation():15b}'[:-3]  # first 12 bits are the wheel, next 3 bits are the between-wheel connections
        count = 0
        for i in range(6):
            if bin_rep[i] == '1' and bin_rep[(i + 1) % 6] == '1' and bin_rep[6 + i] == '1':
                count += 1
        return count

    def connected_vertices(self):
        wheel = self.wheels[1][1]
        center_connected = False
        spoke_connected = [False] * 6
        for i in range(6):
            if (0, i + 1) in wheel.active_edges:
                center_connected = True
                spoke_connected[i] = True
            if (i + 1, (i + 1) % 6 + 1) in wheel.active_edges:
                spoke_connected[i] = True
                spoke_connected[(i + 1) % 6] = True
        for (row1, col1, spoke1), (row2, col2, spoke2) in self.active_btwn_wheel_edges:
            if (row1, col1) == (1, 1):
                spoke_connected[spoke1 - 1] = True
            if (row2, col2) == (1, 1):
                spoke_connected[spoke2 - 1] = True
        v = 0
        if center_connected:
            v += 1
        v += sum(spoke_connected)
        return v


def fig1(save=True):
    '''Generate a figure showing three wheels, all with n=6 spokes and spoke angles 30 degrees, but with different center angles.'''
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    theta_C_values = [0, 60 * DEG, 120 * DEG]
    for ax, theta_C in zip(axes, theta_C_values):
        wheel = Wheel(center=(0, 0), scale=1.0, n=6, angles=[theta_C] + [30 * DEG] * 6)
        wheel.render_wheel(ax, draw_dipoles=True)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"$\\theta_C = {int(theta_C / DEG)}^\\circ$")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p2_1_concept.png"), dpi=300)




def fig2_mechanism_triangular(save=True):
    '''Generate a figure showing 3 different 3x3 triangular grids with n=6 spokes and spoke angles 30 degrees, but with different center angles.'''
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    wheel_grid1 = TriangleWheelTiling(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 0] + [30 * DEG] * 6)
    wheel_grid2 = TriangleWheelTiling(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 50] + [30 * DEG] * 6)
    wheel_grid3 = TriangleWheelTiling(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 100] + [30 * DEG] * 6)

    for ax in axes[1]:
        ax.set_aspect('equal')
        ax.axis('off')

    wheel_grid1.render_grid(axes[1][0], draw_dipoles=True)
    wheel_grid2.render_grid(axes[1][1], draw_dipoles=True)
    wheel_grid3.render_grid(axes[1][2], draw_dipoles=True)
    axes[1][0].set_title(f"Grid with $\\theta_C = 0$° \n$\\beta_1 = {wheel_grid1.beta1()}$")
    axes[1][1].set_title(f"Grid with $\\theta_C = 50$° \n$\\beta_1 = {wheel_grid2.beta1()}$")
    axes[1][2].set_title(f"Grid with $\\theta_C = 100$° \n$\\beta_1 = {wheel_grid3.beta1()}$")

    theta_c_values = np.linspace(0, 180, 200) * DEG
    J_values_axis1 = [frenkel_coupling(theta_c, 30 * DEG, 0, 1) for theta_c in theta_c_values]
    J_values_axis2 = [frenkel_coupling(theta_c, 30 * DEG, 2 * np.pi / 3, 1) for theta_c in theta_c_values]
    J_values_axis3 = [frenkel_coupling(theta_c, 30 * DEG, 4 * np.pi / 3, 1) for theta_c in theta_c_values]
    axes[0][0].plot(theta_c_values / DEG, J_values_axis1, color='blue')
    axes[0][0].plot(theta_c_values / DEG, J_values_axis2, color='orange')
    axes[0][0].plot(theta_c_values / DEG, J_values_axis3, color='green')
    axes[0][0].axhline(0, color='gray', linestyle='--')
    axes[0][0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][0].set_ylabel("$J / K_s$")
    axes[0][0].set_title("Coupling strength vs. center angle for spoke axes")
    axes[0][0].legend(["Axis 1", "Axis 2", "Axis 3"])

    center_angles = np.linspace(0, 180, 200) * DEG
    beta1_values = []
    for theta_c in center_angles:
        grid = TriangleWheelTiling(rows=3, cols=3, separation=3.0, scale=1.0, angles=[theta_c] + [30 * DEG] * 6)
        beta1_values.append(grid.beta1())
    axes[0][1].plot(center_angles / DEG, beta1_values, color='purple')
    axes[0][1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][1].set_ylabel("Independent loops $\\beta_1$")
    axes[0][1].set_title("Grid loop topology vs. center angle")
    axes[0][1].set_ylim(0, max(beta1_values) + 1)

    beta0_values = []
    for theta_c in center_angles:
        grid = TriangleWheelTiling(rows=3, cols=3, separation=3.0, scale=1.0, angles=[theta_c] + [30 * DEG] * 6)
        beta0_values.append(grid.beta0())
    axes[0][2].plot(center_angles / DEG, beta0_values, color='green')
    axes[0][2].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][2].set_ylabel("Connected components $\\beta_0$")
    axes[0][2].set_title("Grid connectivity vs. center angle")

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p2_2_mechanism.png"), dpi=300)

def fig_2_phase_diagram_triangular(save=True):
    '''Phase diagram of $\\beta_0$ and $\\beta_1$ for a single wheel (n=6) vs center and spoke angles.'''
    theta_c_values = np.linspace(0, 360, 200) * DEG
    spoke_angles = np.linspace(0, 360, 200) * DEG
    Theta_C, Spoke_Angle = np.meshgrid(theta_c_values, spoke_angles)
    beta0_values = np.zeros_like(Theta_C, dtype=int)
    beta1_values = np.zeros_like(Theta_C, dtype=int)

    for i in range(Theta_C.shape[0]):
        for j in range(Theta_C.shape[1]):
            wheel = Wheel(center=(0, 0), scale=1.0, n=6, angles=[Theta_C[i, j]] + [Spoke_Angle[i, j]] * 6)
            beta0_values[i, j] = wheel.beta0()
            beta1_values[i, j] = wheel.beta1()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c0 = axes[0].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta0_values, shading='auto', cmap='viridis')
    fig.colorbar(c0, ax=axes[0], label='$\\beta_0$')
    axes[0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0].set_ylabel("Spoke angle (degrees)")
    axes[0].set_title("Phase diagram of $\\beta_0$ for a single 6-wheel")

    c1 = axes[1].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta1_values, shading='auto', cmap='viridis')
    fig.colorbar(c1, ax=axes[1], label='$\\beta_1$')
    axes[1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[1].set_ylabel("Spoke angle (degrees)")
    axes[1].set_title("Phase diagram of $\\beta_1$ for a single 6-wheel")

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p2_2_phase_diagram.png"), dpi=300)

def fig_2_5_phase_diagram_triangular(save=True):
    '''Phase diagram of $\\beta_0$ and $\\beta_1$ for a 3x3 triangular grid vs center and spoke angles.'''
    theta_c_values = np.linspace(0, 360, 200) * DEG
    spoke_angles = np.linspace(0, 360, 200) * DEG
    Theta_C, Spoke_Angle = np.meshgrid(theta_c_values, spoke_angles)
    beta0_values = np.zeros_like(Theta_C, dtype=int)
    beta1_values = np.zeros_like(Theta_C, dtype=int)

    for i in range(Theta_C.shape[0]):
        for j in range(Theta_C.shape[1]):
            grid = TriangleWheelTiling(rows=3, cols=3, separation=3.0, scale=1.0, angles=[Theta_C[i, j]] + [Spoke_Angle[i, j]] * 6)
            beta0_values[i, j] = grid.beta0()
            beta1_values[i, j] = grid.beta1()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c0 = axes[0].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta0_values, shading='auto', cmap='viridis')
    fig.colorbar(c0, ax=axes[0], label='$\\beta_0$')
    axes[0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0].set_ylabel("Spoke angle (degrees)")
    axes[0].set_title("Phase diagram of $\\beta_0$ for a 3x3 triangular grid")

    c1 = axes[1].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta1_values, shading='auto', cmap='viridis')
    fig.colorbar(c1, ax=axes[1], label='$\\beta_1$')
    axes[1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[1].set_ylabel("Spoke angle (degrees)")
    axes[1].set_title("Phase diagram of $\\beta_1$ for a 3x3 triangular grid")

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p2_2_5_phase_diagram.png"), dpi=300)


FIG4_SEP = 3.05

def find_val(theta_c, ns_angle, ew_angle):
    grid = TriangleWheelTiling(rows=3, cols=3, separation=FIG4_SEP, scale=1.0, angles=[theta_c] + [ns_angle, ew_angle] * 3)
    return grid.euler_characteristic_density()

def fig_4_phase_diagram(save=True, show=False, interactive=False, preprocessed=False):
    '''Using the staggered angling, plot 4 figures showing a phase diagram of beta 1 for a 3x3 grid of wheels as a function of the NS angles and EW angles. The 4 figures differ only by center angle.'''
    fig, axes = plt.subplots(2, 2, figsize=(25, 10))
    theta_c_degrees = [0, 10, 20, 30]
    ns_angle_degrees = np.linspace(0, 360, 200)
    ew_angle_degrees = np.linspace(0, 360, 200)
    if preprocessed:
        metadata, panel_values = _load_fig4_cache()
        theta_c_degrees = metadata['theta_c_degrees']
        ns_angle_degrees = np.array(metadata['ns_angle_degrees'])
        ew_angle_degrees = np.array(metadata['ew_angle_degrees'])
    else:
        panel_values = []
        for theta_c_degree in theta_c_degrees:
            theta_c = theta_c_degree * DEG
            NS_angles = ns_angle_degrees * DEG
            EW_angles = ew_angle_degrees * DEG
            NS_grid, EW_grid = np.meshgrid(NS_angles, EW_angles)

            # beta1_values = np.zeros_like(NS_grid, dtype=int)
            # for i in range(NS_grid.shape[0]):
            #     for j in range(NS_grid.shape[1]):
            #         beta1_values[i, j] = find_val(theta_c, NS_grid[i, j], EW_grid[i, j])
            with Pool() as pool:
                args = [(theta_c, ns_angle, ew_angle) for ns_angle in NS_angles for ew_angle in EW_angles]
                beta1_values_flat = pool.starmap(find_val, args)
            beta1_values = np.transpose(np.array(beta1_values_flat).reshape(NS_grid.shape))

            panel_values.append(beta1_values)
        _save_fig4_cache(
            {
                'theta_c_degrees': theta_c_degrees,
                'ns_angle_degrees': ns_angle_degrees.tolist(),
                'ew_angle_degrees': ew_angle_degrees.tolist(),
            },
            panel_values,
        )

    theta_c_values = [degree * DEG for degree in theta_c_degrees]
    ax_to_theta = {}
    NS_grid, EW_grid = np.meshgrid(ns_angle_degrees * DEG, ew_angle_degrees * DEG)
    for ax, angle, beta1_values in zip(axes.flatten(), theta_c_values, panel_values):
        ax.set_xlabel("NS spoke angle (degrees)")
        ax.set_ylabel("EW spoke angle (degrees)")
        ax.set_title("Phase diagram of $\\beta_1$ for a 3x3 grid with $\\theta_C = {}$°".format(angle / DEG))
        ax_to_theta[ax] = angle
        max_beta1 = np.max(beta1_values)
        min_beta1 = np.min(beta1_values)
        c = ax.pcolormesh(NS_grid / DEG, EW_grid / DEG, beta1_values, shading='auto', cmap='viridis', vmin=min_beta1, vmax=max_beta1)
        fig.colorbar(c, ax=ax, label='$\\beta_1$')
    backend_is_interactive = matplotlib.get_backend().lower() != "agg"

    if interactive and backend_is_interactive:
        def on_click(event):
            if event.inaxes not in ax_to_theta:
                return
            if event.xdata is None or event.ydata is None:
                return
            if event.button != 1:
                return
            theta_c = ax_to_theta[event.inaxes]
            ns_angle = event.xdata * DEG
            ew_angle = event.ydata * DEG
            grid = TriangleWheelTiling(
                rows=3,
                cols=3,
                separation=FIG4_SEP,
                scale=1.0,
                angles=[theta_c] + [ns_angle, ew_angle] * 3,
            )
            fig_grid, ax_grid = plt.subplots(figsize=(8, 8))
            grid.render_grid(ax_grid, draw_dipoles=True)
            ax_grid.set_aspect('equal')
            ax_grid.axis('off')
            ax_grid.set_title(
                "Grid view: $\\theta_C = {:.1f}$°, NS = {:.1f}$^\\circ$, EW = {:.1f}$^\\circ$\n$\\beta_0 = {}$, $\\beta_1 = {}$, $\\rho_0 = {}$, $\\rho_1 = {}$".format(
                    theta_c / DEG,
                    event.xdata,
                    event.ydata,
                    grid.beta0(),
                    grid.beta1(),
                    grid.cc_density(),
                    grid.beta1_density(),
                )
            )
            fig_grid.tight_layout()
            fig_grid.show()

        fig.canvas.mpl_connect("button_press_event", on_click)

    if save: # save the figure to the figures directory if specified
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p1_4_phase_diagram.png"), dpi=300)
    if show and backend_is_interactive:
        plt.show()
    elif show and not backend_is_interactive:
        print("Interactive display skipped because matplotlib backend is non-interactive:", matplotlib.get_backend())


if __name__ == "__main__":
    # tiling = TriangleWheelTiling(rows=6, cols=6, separation=3.0, scale=1.0, angles=[240 * DEG] + [30 * DEG] * 6)
    # print("Beta0:", tiling.beta0())
    # print("Beta1:", tiling.beta1())
    # fig, ax = plt.subplots(figsize=(10, 10))
    # tiling.render_grid(ax, draw_dipoles=True)
    # ax.set_aspect('equal')
    # ax.axis('off')
    # plt.tight_layout()
    # plt.savefig(os.path.join(FIGURES_DIR, "fig_triangular.png"), dpi=300)
    # print("Generating figure 1...")
    # fig1(save=True)
    # print("Generating figure 2 mechanism...")
    # fig2_mechanism_triangular(save=True)
    # print("Generating figure 2 phase diagram...")
    # fig_2_phase_diagram_triangular(save=True)
    # print("Generating figure 2.5 phase diagram...")
    # fig_2_5_phase_diagram_triangular(save=True)
    print("Generating figure 4 phase diagram...")
    fig_4_phase_diagram(save=True, show=True, interactive=True, preprocessed=True)
    print("All figures generated.")















