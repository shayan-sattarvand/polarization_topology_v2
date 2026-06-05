import sys
import os
import json
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
if os.environ.get("MPL_HEADLESS", "1") == "1":
    matplotlib.use('Agg')   # headless rendering
else:
    matplotlib.use('TkAgg')
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

from cc_density import DisjointSetUnion

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'square')
os.makedirs(FIGURES_DIR, exist_ok=True)
FIG4_CACHE_DIR = os.path.join(FIGURES_DIR, 'fig4_cache')
os.makedirs(FIG4_CACHE_DIR, exist_ok=True)

K_s = 1.0


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


class Wheel: 
    def __init__(self, center, scale, n=4, angles = None, offset = 0):
        '''Class representing a wheel graph with n spokes. The center is at `center`, 
        the radius is `scale`, and the angles of the center + spokes can be specified by `angles`.
        If `active_edges` is provided, only those edges will be considered active; otherwise, all 
        edges will be active.'''
        self.center = center
        self.scale = scale
        self.n = n
        self.radius = scale
        self.angle_step = 2 * np.pi / n
        self.offset = offset
        self.theta_ij = list(np.arange(n) * self.angle_step)
        self.spokes = np.array([
            (center[0] + self.radius * np.cos(angle + self.offset), 
             center[1] + self.radius * np.sin(angle + self.offset))
            for angle in self.theta_ij
        ])
        self.spoke_angles = angles.copy()
        self.center_angle = self.spoke_angles.pop(0) # remove the first angle, which is for the center node
        active_edges = []
        all_edges = []
        for i in range(self.n): # between center and spokes
            phi_j = self.spoke_angles[i]
            J = frenkel_coupling(self.center_angle, phi_j, self.angle_step * i + self.offset, 1)
            all_edges.append((0, i + 1, J))
            if J < 0:
                active_edges.append((0, i + 1))
        for i in range(self.n): # between adjacent spokes
            theta_ij = np.arctan2(self.spokes[(i+1)%self.n][1] - self.spokes[i][1], self.spokes[(i+1)%self.n][0] - self.spokes[i][0])
            J = frenkel_coupling(self.spoke_angles[i], self.spoke_angles[(i + 1) % self.n], theta_ij, 1)
            all_edges.append((i + 1, (i + 1) % self.n + 1, J))
            if J < 0:
                active_edges.append((i + 1, (i + 1) % self.n + 1))
        self.active_edges = active_edges
        self.all_edges = all_edges
        

    def render_wheel(self, ax, draw_dipoles=False, node_size=100):
        '''Render the wheel graph on the given matplotlib axis `ax`. Active edges are colored green, while inactive edges are colored gray.'''
        for (i, j, _) in self.all_edges:
            if (i, j) in self.active_edges:
                color = '#1a9850'  # bright green
                linewidth = 3
            else:
                color = '#EEEEEE'  # light gray
                linewidth = 1
            if (i, j)[0] == 0:  # spoke edge
                start = self.center
                end = self.spokes[(i, j)[1] - 1]
            else:  # rim edge
                start = self.spokes[(i, j)[0] - 1]
                end = self.spokes[(i, j)[1] - 1]
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth)
        ax.scatter(*self.center, color='blue', s=node_size, zorder=3)  # center node
        ax.scatter(self.spokes[:, 0], self.spokes[:, 1], color='#AAAAAA', s=node_size, zorder=3)  # spoke nodes
        if draw_dipoles:
            for i in range(self.n):
                angle = self.spoke_angles[i]
                dx = 0.2 * np.cos(angle)
                dy = 0.2 * np.sin(angle)
                ax.arrow(self.spokes[i-1][0], self.spokes[i-1][1], dx, dy, head_width=0.05, head_length=0.1, fc='red', ec='red', zorder=4)
            ax.arrow(self.center[0], self.center[1], 0.2 * np.cos(self.center_angle), 0.2 * np.sin(self.center_angle), head_width=0.05, head_length=0.1, fc='red', ec='red', zorder=4)
        ax.set_aspect('equal')
    
    def beta0(self):
        '''Calculate the zeroth Betti number of the wheel graph based on the active edges.'''
        parent = list(range(self.n + 1))  # include center node (0)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for u, v in self.active_edges:
            union(u, v)

        return len({find(i) for i in range(self.n + 1)})

    def beta1(self):
        '''Calculate the first Betti number of the wheel graph based on the active edges.'''
        if not self.active_edges:
            return 0

        active_nodes = set()
        adjacency = {}

        for u, v in self.active_edges:
            active_nodes.add(u)
            active_nodes.add(v)
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

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

        return len(self.active_edges) - len(active_nodes) + components
    
    def binary_representation(self):
        '''return a number which encodes the wheel information in its binary digits.
        the first n digits encode the presence or absence of spoke-center edges, and the next n digits encode the presence or absence of rim edges.'''
        binary = 0
        for i in range(self.n): # center to spoke 
            binary *= 2
            binary += (1 if (0, i + 1) in self.active_edges else 0)
        for i in range(self.n): # rim
            binary *= 2
            binary += (1 if (i + 1, (i + 1) % self.n + 1) in self.active_edges else 0)
        return binary

class SquareWheelGrid:
    def __init__(self, rows, cols, separation=3, scale=1, angles=None):
        '''Class representing a grid of wheels with n=4 spokes. Each wheel is represented by an instance of the `Wheel` class.'''
        self.rows = rows
        self.cols = cols
        self.separation = separation
        self.wheels = []
        for i in range(rows):
            row = []
            for j in range(cols):
                center = (j * separation, i * separation)
                wheel = Wheel(center, scale, 4, angles=angles)
                row.append(wheel)
            self.wheels.append(row)
        btwn_wheel_connections = []
        self.active_btwn_wheel_edges = []
        self.horiz_active = False
        self.vert_active = False
        # Horizontal first
        for i in range(rows):
            for j in range(cols - 1):
                wheel1 = self.wheels[i][j]
                wheel2 = self.wheels[i][j + 1]
                phi1 = wheel1.spoke_angles[0]  # spoke 1 of wheel 1
                phi2 = wheel2.spoke_angles[2]  # spoke 3 of wheel 2
                theta_ij = np.arctan2(wheel2.spokes[2][1] - wheel1.spokes[0][1], wheel2.spokes[2][0] - wheel1.spokes[0][0])
                J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                if J < 0:
                    self.active_btwn_wheel_edges.append(((i, j, 1), (i, j + 1, 3)))  # connect spoke 1 of wheel (i, j) to spoke 3 of wheel (i, j + 1)
                    self.horiz_active = True
                btwn_wheel_connections.append(((i, j, 1), (i, j + 1, 3), J))  # connect spoke 1 of wheel (i, j) to spoke 3 of wheel (i, j + 1)
        # Then vertical
        for i in range(rows - 1):
            for j in range(cols):
                wheel1 = self.wheels[i][j]
                wheel2 = self.wheels[i + 1][j]
                phi1 = wheel1.spoke_angles[1]  # spoke 2 of wheel 1
                phi2 = wheel2.spoke_angles[3]  # spoke 4 of wheel 2
                theta_ij = np.arctan2(wheel2.spokes[3][1] - wheel1.spokes[1][1], wheel2.spokes[3][0] - wheel1.spokes[1][0])
                J = frenkel_coupling(phi1, phi2, theta_ij, 1)
                if J < 0:
                    self.active_btwn_wheel_edges.append(((i, j, 2), (i + 1, j, 4)))  # connect spoke 2 of wheel (i, j) to spoke 4 of wheel (i + 1, j)
                    self.vert_active = True
                btwn_wheel_connections.append(((i, j, 2), (i + 1, j, 4), J))  # connect spoke 2 of wheel (i, j) to spoke 4 of wheel (i + 1, j)
        self.btwn_wheel_connections = btwn_wheel_connections

    
    def render_grid(self, ax, draw_dipoles=False):
        '''Render the grid of wheels on the given matplotlib axis `ax`.'''
        for row in self.wheels:
            for wheel in row:
                wheel.render_wheel(ax, draw_dipoles=draw_dipoles)
        for ((row1, col1, spoke1), (row2, col2, spoke2), _) in self.btwn_wheel_connections:
            wheel1 = self.wheels[row1][col1]
            wheel2 = self.wheels[row2][col2]
            start = wheel1.spokes[spoke1 - 1]
            end = wheel2.spokes[spoke2 - 1]
            if ((row1, col1, spoke1), (row2, col2, spoke2)) in self.active_btwn_wheel_edges:
                color = '#1a9850'  # bright green
                linewidth = 2
            else:
                color = '#CCCCCC'  # light gray
                linewidth = 1
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth)
        ax.set_aspect('equal')

    def render_single_wheel(self, ax, row, col, draw_dipoles=False):
        '''Render a single wheel from the grid on the given matplotlib axis `ax`.'''
        self.wheels[row][col].render_wheel(ax, draw_dipoles=draw_dipoles)
    
    def beta0(self):
        '''Calculate the zeroth Betti number of the grid based on active edges.'''
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
        '''Calculate the first Betti number of the grid based on the active edges.'''
        def global_id(r, c, local_id):
            return (r * self.cols + c) * (self.wheels[0][0].n + 1) + local_id

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
        '''Return a binary representation of the tiling's center wheels. First 8 digits are the wheel representation, next 2 digits are the horizontal and vertical between-wheel connections.'''
        wheel = self.wheels[1][1]
        num = wheel.binary_representation()
        for i in [self.horiz_active, self.vert_active]:
            num *= 2
            if i:
                num += 1
        return num
    
    def cc_density(self, include_lone = True):
        """
        Calculate the CC density for a given binary representation of a configuration.
        
        Parameters:
        binary_representation (str): A string of '0's and '1's representing the configuration.
        
        Returns:
        float: The calculated CC density.
        """
        binary_representation = self.binary_representation()
        dsu = DisjointSetUnion(5)
        inter_wheel = binary_representation % 4
        wheel_rep = binary_representation // 4
        binary_representation = binary_representation // 4
        rim_connections = binary_representation % 16
        binary_representation = binary_representation // 16
        wheel_connections = binary_representation
        for i in range(4):
            if rim_connections & (1 << (3 - i)):
                dsu.union(i + 1, (i + 1) % 4 + 1)
        for i in range(4):
            if wheel_connections & (1 << (3 - i)):
                dsu.union(0, i + 1)
        inter_wheel_new = inter_wheel
        if (inter_wheel & 2 and dsu.query(1,3)) or (inter_wheel & 1 and dsu.query(2,4)):
            path_locs = []
            if inter_wheel & 2 and dsu.query(1,3):
                inter_wheel_new -= 2
                path_locs.append(dsu.find(1))
            if inter_wheel & 1 and dsu.query(2,4):
                inter_wheel_new -= 1
                path_locs.append(dsu.find(2))
            for path_loc in path_locs:
                for i in range(4):
                    loc = dsu.find(i + 1)
                    if loc == path_loc and wheel_rep & (1 << (7 - i)):
                        wheel_rep -= 1 << (7 - i)
                    loc2 = dsu.find((i + 1) % 4 + 1)
                    if loc2 == path_loc and loc == path_loc and wheel_rep & (1 << (3 - i)):
                        wheel_rep -= 1 << (3 - i)
            
        #ugh check zigzags
        if inter_wheel & 3 == 3 and ((dsu.query(1, 2) and dsu.query(3, 4)) or (dsu.query(1, 4) and dsu.query(2, 3))):
            inter_wheel_new = 0
            path_locs = [dsu.find(1), dsu.find(3)]
            for path_loc in path_locs:
                for i in range(4):
                    loc = dsu.find(i + 1)
                    if loc == path_loc and wheel_rep & (1 << (7 - i)):
                        wheel_rep -= 1 << (7 - i)
                    loc2 = dsu.find((i + 1) % 4 + 1)
                    if loc2 == path_loc and loc == path_loc and wheel_rep & (1 << (3 - i)):
                        wheel_rep -= 1 << (3 - i)
        dsu_now = DisjointSetUnion(5)
        for i in range(4):
            if wheel_rep & (1 << (3 - i)):
                dsu_now.union(i + 1, (i + 1) % 4 + 1)
        for i in range(5):
            dsu_now.find(i)
        for i in range(4):
            if wheel_rep & (1 << (7 - i)):
                dsu_now.union(0, i + 1)
        for i in range(5):
            dsu_now.find(i)
        for i in range(2):
            if inter_wheel_new & (1 << (1 - i)):
                dsu_now.union(i + 1, i + 3)
        for i in range(5):
            dsu_now.find(i)
        if include_lone:
            return dsu_now.connected_components_edges() + 5 - self.connected_vertices()
        else:
            return dsu_now.connected_components_edges()

    def beta1_density(self):
        e = sum(1 for i in f'{self.binary_representation():b}' if i == '1')
        v = self.connected_vertices()
        c = self.cc_density(include_lone=False)
        t = self.num_triangles()
        return max(0, e - v + c - t)
    
    def num_triangles(self):
        bin_rep = f'{self.binary_representation():10b}'[:-2]  # first 8 bits are the wheel, next 2 bits are the between-wheel connections
        count = 0
        for i in range(4):
            if bin_rep[i] == '1' and bin_rep[(i + 1) % 4] == '1' and bin_rep[4 + i] == '1':
                count += 1
        return count
    
    def connected_vertices(self):
        wheel = self.wheels[1][1]
        center_connected = False
        spoke_connected = [False] * 4
        for i in range(4):
            if (0, i + 1) in wheel.active_edges:
                center_connected = True
                spoke_connected[i] = True
            if (i + 1, (i + 1) % 4 + 1) in wheel.active_edges:
                spoke_connected[i] = True
                spoke_connected[(i + 1) % 4] = True
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
    '''Generate a figure showing three wheels, all with n=4 spokes and spoke angles 45 degrees, but with different center angles.'''
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5)) # make subplots
    wheel1 = Wheel(center=(0, 0), scale=1.0, n=4, angles=[DEG * 10]+[np.pi/4] * 4) # center angle 10 degrees
    wheel2 = Wheel(center=(0, 0), scale=1.0, n=4, angles=[DEG * 45]+[np.pi/4] * 4) # center angle 45 degrees
    wheel3 = Wheel(center=(0, 0), scale=1.0, n=4, angles=[DEG * 80]+[np.pi/4] * 4) # center angle 80 degrees
    for ax in axes: #set limits and aspect ratio for each subplot
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')
    wheel1.render_wheel(axes[0], draw_dipoles=True) # render each wheel on its respective subplot
    wheel2.render_wheel(axes[1], draw_dipoles=True)
    wheel3.render_wheel(axes[2], draw_dipoles=True)
    axes[0].set_title(f"Wheel with $\\theta_C = 10$° \n$\\beta_0 = {wheel1.beta0()}$, $\\beta_1 = {wheel1.beta1()}$") # set title for each subplot with center angle and Betti numbers
    axes[1].set_title(f"Wheel with $\\theta_C = 45$° \n$\\beta_0 = {wheel2.beta0()}$, $\\beta_1 = {wheel2.beta1()}$")
    axes[2].set_title(f"Wheel with $\\theta_C = 80$° \n$\\beta_0 = {wheel3.beta0()}$, $\\beta_1 = {wheel3.beta1()}$")
    if save: # save the figure to the figures directory if specified
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p1_1_concept.png"), dpi=300)

def fig2_mechanism(save=True):
    '''Generate a figure showing 3 different 3x3 grids of wheels, all with n=4 spokes and spoke angles 45 degrees, but with different center angles.'''
    fig, axes = plt.subplots(2, 3, figsize=(15, 12)) # make subplots
    wheel_grid1 = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 10]+[np.pi/4] * 4) # center angle 10 degrees
    wheel_grid2 = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 45]+[np.pi/4] * 4) # center angle 45 degrees
    wheel_grid3 = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 80]+[np.pi/4] * 4) # center angle 80 degrees
    for ax in axes[1]: #set limits and aspect ratio for each subplot
        ax.set_xlim(-2, 8)
        ax.set_ylim(-2, 8)
        ax.set_aspect('equal')
        ax.axis('off')
    wheel_grid1.render_grid(axes[1][0], draw_dipoles=True) # render each grid on its respective subplot
    wheel_grid2.render_grid(axes[1][1], draw_dipoles=True)
    wheel_grid3.render_grid(axes[1][2], draw_dipoles=True)
    axes[1][0].set_title(f"Grid with $\\theta_C = 10$° \n$\\beta_1 = {wheel_grid1.beta1()}$") # set title for each subplot with center angle and first Betti number
    axes[1][1].set_title(f"Grid with $\\theta_C = 45$° \n$\\beta_1 = {wheel_grid2.beta1()}$")
    axes[1][2].set_title(f"Grid with $\\theta_C = 80$° \n$\\beta_1 = {wheel_grid3.beta1()}$")
    # We're done with bottom row of figure 2, now for the specific graphs in the top row. 
    # first one: graph of coupling strengths as a function of theta_c for the spoke edges of a single wheel
    theta_c_values = np.linspace(0, 90, 100) * DEG
    J_values_NS = [frenkel_coupling(theta_c, np.pi/4, np.pi/2, 1) for theta_c in theta_c_values]
    axes[0][0].plot(theta_c_values / DEG, J_values_NS, color='orange')
    J_values_EW = [frenkel_coupling(theta_c, np.pi/4, 0, 1) for theta_c in theta_c_values]
    axes[0][0].plot(theta_c_values / DEG, J_values_EW, color='blue')
    axes[0][0].axhline(0, color='gray', linestyle='--') # add horizontal line at J=0 for reference
    # add vertical lines at angles where J=0 for each type of spoke edge
    gamma_ns = np.degrees(np.arctan2(np.cos(np.pi/4), 2.0 * np.sin(np.pi/4)))
    gamma_ew = 90 - gamma_ns
    axes[0][0].axvline(gamma_ns, color='grey', linestyle='--')
    axes[0][0].axvline(gamma_ew, color='grey', linestyle='--')
    axes[0][0].text(gamma_ns + 2, 0.5, f"$\\Gamma_{{NS}} \\approx {gamma_ns:.1f}$°", color='grey')
    axes[0][0].text(gamma_ew + 2, 0.5, f"$\\Gamma_{{EW}} \\approx {gamma_ew:.1f}$°", color='grey')
    axes[0][0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][0].set_ylabel("$J / K_s$")
    axes[0][0].set_title("Coupling strength vs. center angle for spoke edges")
    axes[0][0].legend(["Spoke to N/S", "Spoke to E/W"])
    # second one: graph of loop topology for 3x3 grid as a function of center angle, showing how the first Betti number changes with angle.
    center_angles = np.linspace(0, 90, 100) * DEG
    beta1_values = []
    for theta_c in center_angles:
        grid = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[theta_c]+[np.pi/4] * 4)
        beta1_values.append(grid.beta1())
    axes[0][1].plot(center_angles / DEG, beta1_values, color='purple')
    axes[0][1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][1].set_ylabel("Independent loops $\\beta_1$")
    axes[0][1].set_title("Grid loop topology vs. center angle")
    axes[0][1].text(gamma_ns + 2, 8, f"$\\Gamma_{{NS}} \\approx {gamma_ns:.1f}$°", color='grey')
    axes[0][1].text(gamma_ew + 2, 8, f"$\\Gamma_{{EW}} \\approx {gamma_ew:.1f}$°", color='grey')
    axes[0][1].axvline(gamma_ns, color='grey', linestyle='--')
    axes[0][1].axvline(gamma_ew, color='grey', linestyle='--')
    axes[0][1].set_ylim(0, max(beta1_values) + 1)
    # third one: same as second, but beta 0 instead of beta 1. In this case, it should be constant at 1 for all angles, since the grid is always connected.
    beta0_values = []
    for theta_c in center_angles:
        grid = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[theta_c]+[np.pi/4] * 4)
        beta0_values.append(grid.beta0())
    axes[0][2].plot(center_angles / DEG, beta0_values, color='green')
    axes[0][2].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0][2].set_ylabel("Connected components $\\beta_0$")
    axes[0][2].set_title("Grid connectivity vs. center angle")
    axes[0][2].set_ylim(0, 2) # set y limits to show that beta0 is always 1

    if save: # save the figure to the figures directory if specified
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p1_2_mechanism.png"), dpi=300)

def fig_2_phase_diagram(save=True):
    '''Generate a figure showing the phase diagram of the zeroth and first Betti numbers for a single wheel with n=4 spokes as a function of the center angle and the spoke angles.'''
    theta_c_values = np.linspace(0, 360, 500) * DEG
    spoke_angles = np.linspace(0, 360, 500) * DEG
    Theta_C, Spoke_Angle = np.meshgrid(theta_c_values, spoke_angles)
    beta0_values = np.zeros_like(Theta_C, dtype=int)
    beta1_values = np.zeros_like(Theta_C, dtype=int)
    for i in range(Theta_C.shape[0]):
        for j in range(Theta_C.shape[1]):
            wheel = Wheel(center=(0, 0), scale=1.0, n=4, angles=[Theta_C[i, j]] + [Spoke_Angle[i, j]] * 4)
            beta0_values[i, j] = wheel.beta0()
            beta1_values[i, j] = wheel.beta1()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c0 = axes[0].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta0_values, shading='auto', cmap='viridis', vmin=0, vmax=6)
    fig.colorbar(c0, ax=axes[0], label='$\\beta_0$')
    axes[0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0].set_ylabel("Spoke angle (degrees)")
    axes[0].set_title("Phase diagram of $\\beta_0$ for a single wheel")
    c1 = axes[1].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta1_values, shading='auto', cmap='viridis')
    fig.colorbar(c1, ax=axes[1], label='$\\beta_1$')
    axes[1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[1].set_ylabel("Spoke angle (degrees)")
    axes[1].set_title("Phase diagram of $\\beta_1$ for a single wheel")
    if save: # save the figure to the figures directory if specified
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p1_2_phase_diagram.png"), dpi=300)

def fig_2_5_phase_diagram(save=True):
    '''Phase diagram of $\\beta_0$ and $\\beta_1$ for a 3x3 grid vs center and spoke angles.'''
    theta_c_values = np.linspace(0, 360, 200) * DEG
    spoke_angles = np.linspace(0, 360, 200) * DEG
    Theta_C, Spoke_Angle = np.meshgrid(theta_c_values, spoke_angles)
    beta0_values = np.zeros_like(Theta_C, dtype=int)
    beta1_values = np.zeros_like(Theta_C, dtype=int)

    for i in range(Theta_C.shape[0]):
        for j in range(Theta_C.shape[1]):
            grid = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[Theta_C[i, j]] + [Spoke_Angle[i, j]] * 4)
            beta0_values[i, j] = grid.beta0()
            beta1_values[i, j] = grid.beta1()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c0 = axes[0].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta0_values, shading='auto', cmap='viridis')
    fig.colorbar(c0, ax=axes[0], label='$\\beta_0$')
    axes[0].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[0].set_ylabel("Spoke angle (degrees)")
    axes[0].set_title("Phase diagram of $\\beta_0$ for a 3x3 grid")

    c1 = axes[1].pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, beta1_values, shading='auto', cmap='viridis')
    fig.colorbar(c1, ax=axes[1], label='$\\beta_1$')
    axes[1].set_xlabel("Center angle $\\theta_C$ (degrees)")
    axes[1].set_ylabel("Spoke angle (degrees)")
    axes[1].set_title("Phase diagram of $\\beta_1$ for a 3x3 grid")

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p1_2_5_phase_diagram.png"), dpi=300)

def find_val(theta_c, ns_angle, ew_angle):
    grid = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[theta_c] + [ns_angle, ew_angle] * 2)
    val = grid.beta1_density()
    return val

def fig_4_phase_diagram(save=True, show=False, interactive=False, preprocessed=False):
    '''Using the staggered angling, plot 4 figures showing a phase diagram of beta 1 for a 3x3 grid of wheels as a function of the NS angles and EW angles. The 4 figures differ only by center angle.'''
    fig, axes = plt.subplots(2, 2, figsize=(25, 10))
    theta_c_degrees = [10, 45, 80, 90]
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
            #         grid = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[theta_c] + [NS_grid[i, j], EW_grid[i, j]] * 2)
            #         beta1_values[i, j] = grid.beta1_density()
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
            grid = SquareWheelGrid(
                rows=3,
                cols=3,
                separation=3.0,
                scale=1.0,
                angles=[theta_c] + [ns_angle, ew_angle] * 2,
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

    if save and not preprocessed: # save the figure to the figures directory if specified
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p1_4_phase_diagram.png"), dpi=300)
    if show and backend_is_interactive:
        plt.show()
    elif show and not backend_is_interactive:
        print("Interactive display skipped because matplotlib backend is non-interactive:", matplotlib.get_backend())
    
def fig_5_phase_diagram(save=True):
    '''Generate a figure showing sign of NS * sign of EW * delta E for a single wheel with n=4 spokes as a function of the center angle and the spoke angles. This is the quantity that determines whether the grid will have loops or not, since it determines whether the NS and EW edges can be active at the same time.'''
    theta_c_values = np.linspace(0, 90.0, 200) * DEG
    spoke_angles = np.linspace(0, 90.0, 200) * DEG
    Theta_C, Spoke_Angle = np.meshgrid(theta_c_values, spoke_angles)
    sign_product = np.zeros_like(Theta_C, dtype=float)

    for i in range(Theta_C.shape[0]):
        for j in range(Theta_C.shape[1]):
            theta_c = Theta_C[i, j]
            spoke_angle = Spoke_Angle[i, j]
            J_NS = frenkel_coupling(theta_c, spoke_angle, np.pi/2, 1)
            J_EW = frenkel_coupling(theta_c, spoke_angle, 0, 1)
            M = symmetric_block(theta_c, spoke_angle, K_s)
            evals_sym = np.linalg.eigvalsh(M)
            delta_E = float(evals_sym[1] - evals_sym[0])
            sign_product[i, j] = float(np.sign(J_NS)) * float(np.sign(J_EW)) * delta_E

    max_val = np.max(sign_product)
    min_val = np.min(sign_product)
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(Theta_C / DEG, Spoke_Angle / DEG, sign_product, shading='auto', cmap='PRGn', vmin=min_val-1, vmax=max_val+1)
    fig.colorbar(c, ax=ax, label='$sign(J_{NS}) \\cdot sign(J_{EW}) \\cdot \\Delta E$')
    ax.set_xlabel("Center angle $\\theta_C$ (degrees)")
    ax.set_ylabel("Spoke angle (degrees)")
    ax.set_title("Phase diagram of $sign(J_{NS}) \\cdot sign(J_{EW}) \\cdot \\Delta E$ for a single wheel")
    if save: # save the figure to the figures directory if specified
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p1_5_phase_diagram.png"), dpi=300)


if __name__ == "__main__":
    # print("Generating figure 1...")
    # fig1(save=True)
    # print("Generating figure 2 mechanism...")
    # fig2_mechanism(save=True)
    # print("Generating figure 2 phase diagram...")
    # fig_2_phase_diagram(save=True)
    # print("Generating figure 2.5 phase diagram...")
    # fig_2_5_phase_diagram(save=True)
    print("Generating figure 4 phase diagram...")
    fig_4_phase_diagram(save=True, show=True, interactive=True, preprocessed=False)
    # print("Generating figure 5 phase diagram...")
    # fig_5_phase_diagram(save=True)
