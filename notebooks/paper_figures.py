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
    '''Figure 1: 3 wheel graphs for n=3,4,6'''
    fig = plt.figure(layout="constrained", figsize=(24, 16))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    gs00 = fig.add_subplot(gs[0, 0])
    gs01 = fig.add_subplot(gs[0, 1])
    gs02 = fig.add_subplot(gs[0, 2])
    w3 = Wheel((0, 0), 1, 3, [60 * DEG] + [120 * DEG] * 3, 0)
    w4 = Wheel((0, 0), 1, 4, [45 * DEG] + [90 * DEG] * 4, 0)
    w6 = Wheel((0, 0), 1, 6, [30 * DEG] + [60 * DEG] * 6, 0)
    w3.render_wheel(gs00, True)
    w4.render_wheel(gs01, True)
    w6.render_wheel(gs02, True)
    gs00.set_title('Wheel graph with $n = 3$', fontsize=36, fontweight='bold')
    gs01.set_title('Wheel graph with $n = 4$', fontsize=36, fontweight='bold')
    gs02.set_title('Wheel graph with $n = 6$', fontsize=36, fontweight='bold')
    for i in [gs00, gs01, gs02]:
        i.set_aspect('equal')
        i.set_box_aspect(1)
        i.set_xlim(-1.275, 1.275)
        i.set_ylim(-1.275, 1.275)
        i.set_anchor('C')
        i.axis('off')

    plt.savefig(os.path.join(FIGURES_DIR, 'fig_1.png'), dpi=300)


def fig_2():
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
    hex = fig_hexagonal.HexagonalWheelTiling(6, 3, angles=[60 * DEG] + [120 * DEG] * 3)
    hex.render_grid(ax10, True)
    ax10.text(
        0.5, 0.96,
        f"$\\rho_\\chi = {hex.euler_characteristic_density():.2f}$, $\\rho_0 = {hex.cc_density():.2f}$, $\\rho_1 = {hex.beta1_density():.2f}$",
        fontsize=18,
        ha="center",
        va="bottom",
        transform=ax10.transAxes
    )
    ax10.set_title('Hexagonal tiling', fontsize=36, fontweight='bold')
    square = fig_square.SquareWheelGrid(5, 4, angles=[45 * DEG] + [90 * DEG] * 4)
    square.render_grid(ax11, True)
    ax11.text(
            0.5, 0.96,
            f"$\\rho_\\chi = {square.euler_characteristic_density():.2f}$, $\\rho_0 = {square.cc_density():.2f}$, $\\rho_1 = {square.beta1_density():.2f}$",
            fontsize=18,
            ha="center",
            va="bottom",
            transform=ax11.transAxes
        )
    ax11.set_title('Square tiling', fontsize=36, fontweight='bold')
    triangle = fig_triangular.TriangleWheelTiling(6, 3, angles=[30 * DEG] + [60 * DEG] * 6)
    triangle.render_grid(ax12_left, True)
    ax12_left.text(
            0.5, 0.96,
            f"$\\rho_\\chi = {triangle.euler_characteristic_density():.2f}$, $\\rho_0 = {triangle.cc_density():.2f}$, $\\rho_1 = {triangle.beta1_density():.2f}$",
            fontsize=18,
            ha="center",
            va="bottom",
            transform=ax12_left.transAxes
        )
    ax12_left.set_title('Triangular tiling', fontsize=36, fontweight='bold')
    kagome = fig_kagome.KagomeWheelTiling(12, 3, angles=[30 * DEG] + [60 * DEG] * 6)
    kagome.render_grid(ax12_right, True)
    ax12_right.text(
            0.5, 0.96,
            f"$\\rho_\\chi = {kagome.euler_characteristic_density():.2f}$, $\\rho_0 = {kagome.cc_density():.2f}$, $\\rho_1 = {kagome.beta1_density():.2f}$",
            fontsize=18,
            ha="center",
            va="bottom",
            transform=ax12_right.transAxes
        )
    ax12_right.set_title('Kagome tiling', fontsize=36, fontweight='bold')
    for i in [ax10, ax11, ax12_left, ax12_right]:
        i.axis('off')
    
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_2.png'), dpi=300)

def fig_4():
    fig = plt.figure(layout = "constrained", figsize=(12, 12))
    gs = fig.add_gridspec(4, 4, height_ratios = [1, 1, 1, 1])
    subplots = [[]]
    for i in range(4):
        row = []
        for j in range(4):
            row.append(fig.add_subplot(gs[i, j]))
        subplots.append(row)

    # load in the phase diagrams from the csv's to save time
    for loc, path in zip(list(range(4)), ['square', 'triangular', 'kagome', 'hexagonal']):

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
        if loc == 0:
            ax.plot(200, 250, 'ro', ms=4)
            ax.annotate('A', xy=(210, 250), color='red')
            ax.plot(70, 324, 'ro', ms=4)
            ax.annotate('B', xy=(80, 324), color='red')
            ax.plot(235, 34, 'ro', ms=4)
            ax.annotate('C', xy=(245, 34), color='red')
        if loc == 1:
            ax.plot(270, 270, 'ro', ms=4)
            ax.annotate('A', xy=(280, 270), color='red')
            ax.plot(300, 125, 'ro', ms=4)
            ax.annotate('B', xy=(310, 125), color='red')
            ax.plot(214, 263, 'ro', ms=4)
            ax.annotate('C', xy=(224, 263), color='red')
        if loc == 2:
            ax.plot(270, 270, 'ro', ms=4)
            ax.annotate('A', xy=(280, 270), color='red')
            ax.plot(300, 125, 'ro', ms=4)
            ax.annotate('B', xy=(310, 125), color='red')
            ax.plot(214, 263, 'ro', ms=4)
            ax.annotate('C', xy=(224, 263), color='red')
        if loc == 3:
            ax.plot(109.2, 69.2, 'ro', ms=4)
            ax.annotate('A', xy=(119.2, 69.2), color='red')
            ax.plot(244.5, 130.7, 'ro', ms=4)
            ax.annotate('B', xy=(254.5, 130.7), color='red')
            ax.plot(131, 176.5, 'ro', ms=4)
            ax.annotate('C', xy=(141, 176.5), color='red')
            


    for row, lattice in zip(range(1, 5), ['square ', 'triangular ', 'Kagome ', 'hexagonal ']):
        for col, phase in zip(range(1, 5), ['Fully connected ', 'Layered ', 'Sparsely-connected ']):
            ax = subplots[row][col]
            ns = 25
            l = {}
            if lattice == 'square ':
                grid = fig_square.SquareWheelGrid(3, 3, angles=[0] + [[200 * DEG, 250 * DEG], [70 * DEG, 324 * DEG], [235 * DEG, 34 * DEG]][col - 1] * 2)
            if lattice == 'triangular ':
                grid = fig_triangular.TriangleWheelTiling(3, 3, separation=3.05, angles=[0] + [[270 * DEG, 270 * DEG], [300 * DEG, 125 * DEG], [214 * DEG, 263 * DEG]][col - 1] * 3)
            if lattice == 'Kagome ':
                grid = fig_kagome.KagomeWheelTiling(12, 3, separation=3.05, angles=[0] + [[270 * DEG, 270 * DEG], [300 * DEG, 125 * DEG], [214 * DEG, 263 * DEG]][col - 1] * 3)
                ns = 12
                l = {'linewidth': (1, 3)}
            if lattice == 'hexagonal ':
                grid = fig_hexagonal.HexagonalWheelTiling(6, 4, separation=3.02, angles=[0] + [[109.2 * DEG, 69.2 * DEG], [244.5 * DEG, 130.7 * DEG], [131 * DEG, 176.5 * DEG]][col-1] * 3)
            ax.set_title(phase + lattice + 'lattice ('+['', 'A', 'B', 'C'][col]+')', fontsize=9, fontweight='bold')
            ax.axis('off')
            grid.render_grid(ax, True, node_size=ns, **l)
            ax.text(
                0.5, -0.2,
                f"$\\rho_\\chi = {grid.euler_characteristic_density():.2f}$, $\\rho_0 = {grid.cc_density():.2f}$, $\\rho_1 = {grid.beta1_density():.2f}$",
                            fontsize=8,
                            ha="center",
                            va="bottom",
                            transform=ax.transAxes
            )
            
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_4.png'), dpi=300)

def finding_fig_5():
    while True:
        angles = (np.random.uniform(0, 360, 5) * DEG).tolist()
        grid = fig_square.SquareWheelGrid(5, 5, angles=angles)
        if grid.cc_density() == 0:
            continue # we're looking for sparse
        grid.render_grid(plt.gca(), True, node_size=50)
        plt.axis('off')
        plt.show()
        save = input('Save figure? (y/n): ')
        if save.lower() == 'y':
            print(angles)
        plt.cla()

def fig_5():
    data = [ #connected, layered, sparse
        [2.0640089550470275, 2.7188247771847087, 2.1061547055532284, 0.4336871863070836, 2.2201754873679818],
        [3.4667595321417966, 4.04665914202089, 6.161674425524274, 2.7303146029232193, 0.15432926951351605],
        [5.598134648805069, 4.281450746115639, 4.255679680582351, 0.010301181378933403, 4.819290780319332], 
        [1.703139775941826, 4.521955228932793, 4.643514420800798, 0.27780676576839114, 2.036275762609642],
        [2.701808467306757, 4.97965148424765, 2.4523839788438404, 4.042872700398317, 3.4203660144934607],
        [0.3668007926648161, 3.460828135155743, 3.144397641301067, 1.6196549562403668, 0.6078518381542205],
        [1.1205781499430654, 1.4333733986427475, 1.2956769183634305, 4.276789925314377, 3.17465174788541],
        [1.4939094928746095, 2.7869111467524483, 4.005421609490494, 5.44798914957543, 3.598112578595217], 
        [4.211720492087988, 3.5302188172745637, 3.378575310767112, 0.5339011306234152, 0.19335705513097326]
    ]
    fig = plt.figure(layout = "constrained", figsize=(12, 12))
    gs = fig.add_gridspec(4, 3, height_ratios = [0.1, 1, 1, 1])
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i + 1, j])
            angles = data[j * 3 + i]
            grid = fig_square.SquareWheelGrid(5, 5, angles=angles)
            grid.render_grid(ax, True, node_size=50)
            ax.set_title(f'Angles: {[round(x / DEG) for x in angles]}, $\\rho_\\chi = {grid.euler_characteristic_density():.2f}$', fontsize=8)
            # ax.axis('off')
    a, b, c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i in [a, b, c]:
        i.axis('off')
    a.text(0.5, 0.5, 'Connected', ha='center', va='center', fontsize=14, fontweight='bold', transform=a.transAxes)
    b.text(0.5, 0.5, 'Layered', ha='center', va='center', fontsize=14, fontweight='bold', transform=b.transAxes)
    c.text(0.5, 0.5, 'Sparse', ha='center', va='center', fontsize=14, fontweight='bold', transform=c.transAxes)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_5.png'), dpi=300)

def fig_6():
    fig = plt.figure(layout = "constrained", figsize=(9, 6))
    gs = fig.add_gridspec(2, 2, height_ratios = [1, 1])
    ax_left = fig.add_subplot(gs[0,0])
    ax_right = fig.add_subplot(gs[0,1])
    ax_left.axis('off')
    ax_right.axis('off')
    no_defect = fig_square.SquareWheelGrid(3, 5, angles=[0]+[180*DEG,180*DEG]*2)
    no_defect.render_grid(ax_left, linewidth=(1,5), node_size=50)
    with_defect=fig_square.SquareWheelGrid(3, 5, angles=[0]+[180*DEG,180*DEG]*2, defects=[[(1,2), [0]+[278*DEG,212*DEG]*2]])
    with_defect.render_grid(ax_right, linewidth=(1,5), node_size=50)
    ax_left.set_title('Example square tiling', fontweight='bold')
    ax_right.set_title('Same tiling with defect', fontweight='bold')
    ax_left.text(0.5, 0.97, f'$\\beta_0 = {no_defect.beta0()}$, $\\beta_1 = {no_defect.beta1()}$', ha='center',va='bottom',transform=ax_left.transAxes)
    ax_right.text(0.5, 0.97, f'$\\beta_0 = {with_defect.beta0()}$, $\\beta_1 = {with_defect.beta1()}$', ha='center',va='bottom',transform=ax_right.transAxes)
    #show eigenvalue distribution of L1 for them
    no_defect_dist = no_defect.l1_eig_dist()
    with_defect_dist = with_defect.l1_eig_dist()
    eig_fig = fig.add_subplot(gs[1,0:2])
    eig_fig.set_title('Distribution of $L_1$ Laplacian Eigenvalues')
    eig_fig.grid(True)
    for val, occurence in zip(*no_defect_dist):
        eig_fig.plot([val, val], [0, occurence], color='#6666FF', alpha=0.5)
    eig_fig.scatter(no_defect_dist[0], no_defect_dist[1], color='#6666FF', label = 'No Defect Eigenvalues')
    for val, occurence in zip(*with_defect_dist):
        eig_fig.plot([val, val], [0, occurence], color='#FF6666', alpha=0.5)
    eig_fig.scatter(with_defect_dist[0], with_defect_dist[1], color='#FF6666', alpha=0.8, label = 'With Defect Eigenvalues')
    eig_fig.legend()
    eig_fig.set_xlabel("Eigenvalue")
    eig_fig.set_ylabel("Multiplicity")
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_6.png'), dpi=300)

def fig_7(redfield=False):
    fig = plt.figure(layout = "constrained", figsize=(12, 9))
    gs = fig.add_gridspec(3,4)
    fig_w1 = fig.add_subplot(gs[0,0])
    fig_w2 = fig.add_subplot(gs[1,0])
    fig_w3 = fig.add_subplot(gs[2,0])
    w1_spec = fig.add_subplot(gs[0,1])
    w2_spec = fig.add_subplot(gs[1,1])
    w3_spec = fig.add_subplot(gs[2,1])
    w1_2des = fig.add_subplot(gs[0,2])
    w2_2des = fig.add_subplot(gs[1,2])
    w3_2des = fig.add_subplot(gs[2,2])
    w1_anis = fig.add_subplot(gs[0,3])
    w2_anis = fig.add_subplot(gs[1,3])
    w3_anis = fig.add_subplot(gs[2,3])
    w1 = fig_square.Wheel((0,0), 1, 4, [45 * DEG] + [90 * DEG] * 4, 0)
    w2 = fig_square.Wheel((0,0), 1, 4, [45 * DEG] + [45 * DEG, 60 * DEG] * 2, 0)
    w3 = fig_square.Wheel((0,0), 1, 4, [45 * DEG] * 5, 0)
    w1.render_wheel(fig_w1, True)
    w2.render_wheel(fig_w2, True)
    w3.render_wheel(fig_w3, True)
    for ax in [fig_w1, fig_w2, fig_w3]:
        ax.set_aspect('equal')
        ax.set_box_aspect(1)
        ax.set_xlim(-1.275, 1.275)
        ax.set_ylim(-1.275, 1.275)
        ax.set_anchor('C')
        ax.axis('off')
    print('Plotting linear absorption...')
    for ax, w in zip([w1_spec, w2_spec, w3_spec], [w1, w2, w3]):
        print(f'Plotting linear absorption for wheel with angles: {[angle / DEG for angle in w.spoke_angles]}')
        w.plot_linear_absorption(ax, zoom_factor = 0.2, redfield=redfield)
    print('Plotting 2DES spectra...')
    for ax, w in zip([w1_2des, w2_2des, w3_2des], [w1, w2, w3]):
        print(f'Plotting 2DES spectra for wheel with angles: {[angle / DEG for angle in w.spoke_angles]}')
        w.plot_2des(ax, zoom_factor = 0.3, log=False, redfield=redfield)
    print('Plotting 2DES anisotropy...')
    for ax, w in zip([w1_anis, w2_anis, w3_anis], [w1, w2, w3]):
        print(f'Plotting 2DES anisotropy for wheel with angles: {[angle / DEG for angle in w.spoke_angles]}')
        w.plot_2des_anisotropy(ax, zoom_factor = 0.3, redfield=redfield)
    plt.savefig(os.path.join(FIGURES_DIR, f'fig_7{'' if not redfield else '_redfield'}.png'), dpi=300)

def test_2des():
    w = fig_square.Wheel((0,0), 1, 4, [45 * DEG] + [90 * DEG] * 4, 0)
    w.plot_2des(plt.gca(), log=False, redfield=True)
    plt.savefig(os.path.join(FIGURES_DIR, f'test_2des.png'), dpi=300)

if __name__ == '__main__':
    # fig_1()
    # fig_2()
    # fig_4()
    # finding_fig_5()
    fig_5()
    # fig_6()
    # test_2des()
    #fig_7()







