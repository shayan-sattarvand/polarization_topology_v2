import sys
import os
import numpy as np
import json


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')   # headless rendering
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
import matplotlib.pyplot as plt

from fig_square import Wheel
from cc_density import DisjointSetUnion, print_bin

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'paper')
os.makedirs(FIGURES_DIR, exist_ok=True)

from phase_diagram import DEG

import fig_hexagonal
import fig_kagome
import fig_square
import fig_triangular


def fig_1():
    fig = plt.figure(layout="constrained", figsize=(24, 16))

    # Main 2x3 grid
    gs = fig.add_gridspec(2, 3, width_ratios=[1,1,2])

    # Normal subplots
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    top_right = gs[0, 2].subgridspec(1, 3, width_ratios=[0.5, 1, 0.5])
    ax02 = fig.add_subplot(top_right[0, 1])

    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    # Split bottom-right subplot into 2 sub-columns
    subgs = gs[1, 2].subgridspec(1, 2, width_ratios=[1, 1.5])

    ax12_left = fig.add_subplot(subgs[0, 0])
    ax12_right = fig.add_subplot(subgs[0, 1])

    
    w3 = Wheel((0,0), 1, 3, [60 * DEG] + [120 * DEG] * 3, 0)
    w4 = Wheel((0,0), 1, 4, [45 * DEG] + [90 * DEG] * 4, 0)
    w6 = Wheel((0,0), 1, 6, [30 * DEG] + [60 * DEG] * 6, 0)
    w3.render_wheel(ax00, True)
    w4.render_wheel(ax01, True)
    w6.render_wheel(ax02, True)
    ax00.set_title('Wheel graph with $n = 3$', fontsize=36, fontweight='bold')
    ax01.set_title('Wheel graph with $n = 4$', fontsize=36, fontweight='bold')
    ax02.set_title('Wheel graph with $n = 6$', fontsize=36, fontweight='bold')
    for i in [ax00, ax01, ax02]:
        i.set_aspect('equal')
        i.set_box_aspect(1)
        i.set_xlim(-1.275, 1.275)
        i.set_ylim(-1.275, 1.275)
        i.set_anchor('C')
        i.axis('off')
    fig_hexagonal.HexagonalWheelTiling(6, 3, angles=[60 * DEG] + [120 * DEG] * 3).render_grid(ax10, True)
    ax10.set_title('Hexagonal tiling', fontsize=36, fontweight='bold')
    fig_square.SquareWheelGrid(5, 4, angles=[45 * DEG] + [90 * DEG] * 4).render_grid(ax11, True)
    ax11.set_title('Square tiling', fontsize=36, fontweight='bold')
    fig_triangular.TriangleWheelTiling(6, 3, angles=[30 * DEG] + [60 * DEG] * 6).render_grid(ax12_left, True)
    ax12_left.set_title('Triangular tiling', fontsize=36, fontweight='bold')
    fig_kagome.KagomeWheelTiling(12, 3, angles=[30 * DEG] + [60 * DEG] * 6).render_grid(ax12_right, True)
    ax12_right.set_title('Kagome tiling', fontsize=36, fontweight='bold')
    for i in [ax10, ax11, ax12_left, ax12_right]:
        i.axis('off')
    
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_1.png'), dpi=300)

def fig_2():
    fig = plt.figure(layout = "constrained", figsize=(12, 9))
    gs = fig.add_gridspec(4, 4, height_ratios = [0.3, 1, 1, 1])
    subplots = [[]]
    for i in range(3):
        row = []
        for j in range(4):
            row.append(fig.add_subplot(gs[i + 1, j]))
        subplots.append(row)

    # load in the phase diagrams from the csv's to save time
    for loc, path in zip(list(range(3)), ['square', 'triangular', 'kagome']):

        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'figures', path, 'fig4_cache', 'metadata.json')
        panel_path = os.path.join(os.path.dirname(__file__), '..', 'figures', path, 'fig4_cache', 'panel_0.csv')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        panel_values = np.loadtxt(panel_path, delimiter=',', dtype=float)
        ax = subplots[loc + 1][0]
        ns_angle_degrees = np.array(metadata['ns_angle_degrees'])
        ew_angle_degrees = np.array(metadata['ew_angle_degrees'])
        NS_grid, EW_grid = np.meshgrid(ns_angle_degrees * DEG, ew_angle_degrees * DEG)

        ax.set_xlabel("Even spoke angle (degrees)")
        ax.set_ylabel("Odd spoke angle (degrees)")
        ax.set_title(path.capitalize() + ' lattice topological phase diagram', fontweight='bold')
        min_val = np.min(panel_values)
        max_val = np.max(panel_values)
        c = ax.pcolormesh(NS_grid / DEG, EW_grid / DEG, panel_values, shading='auto', cmap='viridis', vmin=min_val, vmax=max_val)
        fig.colorbar(c, ax=ax, label='$\\rho_1$')
    
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_2.png'), dpi=300)
    


if __name__ == '__main__':
    fig_2()







