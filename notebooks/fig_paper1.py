
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

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

K_s = 1.0


class Wheel: 
    def __init__(self, center, scale, n=4, active_edges=None, angles = None):
        '''Class representing a wheel graph with n spokes. The center is at `center`, 
        the radius is `scale`, and the angles of the center + spokes can be specified by `angles`.
        If `active_edges` is provided, only those edges will be considered active; otherwise, all 
        edges will be active.'''
        self.center = center
        self.scale = scale
        self.n = n
        self.radius = scale
        self.angle_step = 2 * np.pi / n
        self.angles = np.arange(n) * self.angle_step
        self.spokes = np.array([
            (center[0] + self.radius * np.cos(angle), 
             center[1] + self.radius * np.sin(angle))
            for angle in self.angles
        ])
        self.all_edges = [(0, i) for i in range(1, n+1)] + [(i, (i % n) + 1) for i in range(1, n+1)]
        if active_edges is None:
            self.active_edges = self.all_edges
        else:
            self.active_edges = active_edges
        self.angles = angles if angles is not None else [np.pi / 2] * (n + 1)

    def calc_active_edges(self):
        '''Calculate the active edges based on the angles. Edges are active if coupling is negative.'''
        active_edges = []
        center_angle = self.angles[0]
        for i in range(1, self.n + 1):
            spoke_angle = self.angles[i]
            J = frenkel_coupling(center_angle, spoke_angle, 2 * np.pi * (i - 1) / self.n, 1)
            if J < 0:
                active_edges.append((0, i))
        for i in range(self.n):
            J = frenkel_coupling(self.angles[i], self.angles[(i + 1) % self.n], np.arctan2(self.spokes[(i+1)%self.n][0] - self.spokes[i][0], self.spokes[(i+1)%self.n][1] - self.spokes[i][1]), 1)
            if J < 0:
                active_edges.append((i + 1, (i + 1) % self.n + 1))
        self.active_edges = active_edges

    def render_wheel(self, ax, draw_dipoles=False):
        '''Render the wheel graph on the given matplotlib axis `ax`. Active edges are colored green, while inactive edges are colored gray.'''
        for edge in self.all_edges:
            if edge in self.active_edges:
                color = '#1a9850'  # bright green
                linewidth = 2
            else:
                color = '#CCCCCC'  # light gray
                linewidth = 1
            if edge[0] == 0:  # spoke edge
                start = self.center
                end = self.spokes[edge[1] - 1]
            else:  # rim edge
                start = self.spokes[edge[0] - 1]
                end = self.spokes[edge[1] - 1]
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth)
        ax.scatter(*self.center, color='blue', s=100, zorder=3)  # center node
        ax.scatter(self.spokes[:, 0], self.spokes[:, 1], color='#aaaaaa', s=100, zorder=3)  # spoke nodes
        if draw_dipoles:
            for i in range(1, self.n + 1):
                angle = self.angles[i]
                dx = 0.2 * np.cos(angle)
                dy = 0.2 * np.sin(angle)
                ax.arrow(self.spokes[i-1][0], self.spokes[i-1][1], dx, dy, head_width=0.05, head_length=0.1, fc='red', ec='red', zorder=4)
            ax.arrow(self.center[0], self.center[1], 0.2 * np.cos(self.angles[0]), 0.2 * np.sin(self.angles[0]), head_width=0.05, head_length=0.1, fc='red', ec='red', zorder=4)
        ax.set_aspect('equal')
    
    def beta1(self):
        '''Calculate the first Betti number of the wheel graph based on the active edges.'''
        n_active_edges = len(self.active_edges)
        n_nodes = self.n + 1  # center + spokes
        beta0 = 1  # wheel graph is always connected
        beta1 = n_active_edges - n_nodes + beta0
        return beta1

    def beta0(self):
        '''Calculate the zeroth Betti number of the wheel graph based on the active edges.'''
        return 1  # wheel graph is always connected

class SquareWheelGrid:
    def __init__(self, rows, cols, separation, scale, angles=None):
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
        # Horizontal first
        for i in range(rows):
            for j in range(cols - 1):
                btwn_wheel_connections.append(((i, j, 1), (i, j + 1, 3)))  # connect spoke 1 of wheel (i, j) to spoke 3 of wheel (i, j + 1)
        # Then vertical
        for i in range(rows - 1):
            for j in range(cols):
                btwn_wheel_connections.append(((i, j, 2), (i + 1, j, 4)))  # connect spoke 2 of wheel (i, j) to spoke 4 of wheel (i + 1, j)
        self.btwn_wheel_connections = btwn_wheel_connections
        self.active_btwn_wheel_edges = []
    
    def calculate_active_edges(self):
        '''Calculate the active edges for all wheels in the grid.'''
        for row in self.wheels:
            for wheel in row:
                wheel.calc_active_edges()
        for ((row1, col1, spoke1), (row2, col2, spoke2)) in self.btwn_wheel_connections:
            wheel1 = self.wheels[row1][col1]
            wheel2 = self.wheels[row2][col2]
            angle1 = wheel1.angles[spoke1]
            angle2 = wheel2.angles[spoke2]
            J = frenkel_coupling(angle1, angle2, np.arctan2(wheel2.spokes[spoke2 - 1][0] - wheel1.spokes[spoke1 - 1][0], wheel2.spokes[spoke2 - 1][1] - wheel1.spokes[spoke1 - 1][1]), 1)
            if J < 0:
                self.active_btwn_wheel_edges.append(((row1, col1, spoke1), (row2, col2, spoke2)))
    
    def render_grid(self, ax, draw_dipoles=False):
        '''Render the grid of wheels on the given matplotlib axis `ax`.'''
        for row in self.wheels:
            for wheel in row:
                wheel.render_wheel(ax, draw_dipoles=draw_dipoles)
        for ((row1, col1, spoke1), (row2, col2, spoke2)) in self.btwn_wheel_connections:
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
    
    def beta1(self):
        '''Calculate the first Betti number of the grid based on the active edges.'''
        btwn_holes = (self.rows - 1) * (self.cols - 1)  # holes between wheels
        wheel_holes = sum(wheel.beta1() for row in self.wheels for wheel in row)  # holes within wheels
        return btwn_holes + wheel_holes

def fig1(save=True):
    '''Generate a figure showing three wheels, all with n=4 spokes and spoke angles 45 degrees, but with different center angles.'''
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5)) # make subplots
    wheel1 = Wheel(center=(0, 0), scale=1.0, n=4, angles=[DEG * 10]+[np.pi/4] * 4) # center angle 10 degrees
    wheel2 = Wheel(center=(0, 0), scale=1.0, n=4, angles=[DEG * 45]+[np.pi/4] * 4) # center angle 45 degrees
    wheel3 = Wheel(center=(0, 0), scale=1.0, n=4, angles=[DEG * 80]+[np.pi/4] * 4) # center angle 80 degrees
    wheel1.calc_active_edges() # calculate active edges for each
    wheel2.calc_active_edges()
    wheel3.calc_active_edges()
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

def fig2(save=True):
    '''Generate a figure showing 3 different 3x3 grids of wheels, all with n=4 spokes and spoke angles 45 degrees, but with different center angles.'''
    fig, axes = plt.subplots(2, 3, figsize=(15, 12)) # make subplots
    wheel_grid1 = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 10]+[np.pi/4] * 4) # center angle 10 degrees
    wheel_grid2 = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 45]+[np.pi/4] * 4) # center angle 45 degrees
    wheel_grid3 = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 80]+[np.pi/4] * 4) # center angle 80 degrees
    wheel_grid1.calculate_active_edges() # calculate active edges for each grid
    wheel_grid2.calculate_active_edges()
    wheel_grid3.calculate_active_edges()
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







    if save: # save the figure to the figures directory if specified
        plt.savefig(os.path.join(FIGURES_DIR, "fig_p1_2_mechanism.png"), dpi=300)


if __name__ == "__main__":
    # # Example usage of the Wheel class
    # wheel = Wheel(center=(0, 0), scale=1.0, n=4, angles=[DEG * 10]+[np.pi/4] * 4)
    # print("Spoke positions:", wheel.spokes)
    # print("Active edges:", wheel.active_edges)
    # print("Calculating active edges based on angles...")
    # wheel.calc_active_edges()
    # print("Updated active edges:", wheel.active_edges)
    # fig, ax = plt.subplots()
    # wheel.render_wheel(ax, draw_dipoles=True)
    # plt.savefig(os.path.join(FIGURES_DIR, "wheel_example.png"), dpi=300)

    # #Test wheel grid now
    # grid = SquareWheelGrid(rows=3, cols=3, separation=3.0, scale=1.0, angles=[DEG * 45]+[np.pi/4] * 4)
    # grid.calculate_active_edges()
    # fig, ax = plt.subplots()
    # grid.render_single_wheel(ax, row=1, col=1, draw_dipoles=True)
    # plt.savefig(os.path.join(FIGURES_DIR, "wheel_grid_single.png"), dpi=300)
    # fig, ax = plt.subplots()
    # grid.render_grid(ax, draw_dipoles=True)
    # plt.savefig(os.path.join(FIGURES_DIR, "wheel_grid_full.png"), dpi=300)
    # print("Grid beta1:", grid.beta1())
    fig1(save=True)
    fig2(save=True)
