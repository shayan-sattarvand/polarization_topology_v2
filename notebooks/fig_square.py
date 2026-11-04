import sys
import os
import json
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
from scipy.linalg import expm, eig

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
    lorentzian,
)

from cc_density import DisjointSetUnion

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'square')
os.makedirs(FIGURES_DIR, exist_ok=True)
FIG4_CACHE_DIR = os.path.join(FIGURES_DIR, 'fig4_cache')
os.makedirs(FIG4_CACHE_DIR, exist_ok=True)

K_s = 1.0 # 1.0 = 100 inverse cm, 2.0 = 200 inverse cm, etc. This is the energy scale of the Hamiltonian.
NEW_J_CONDITION = True
J_SQUARED_THRESHOLD = 0.3

_TWO_DES_WORKER_WHEEL = None


def _init_two_des_worker(wheel):
    global _TWO_DES_WORKER_WHEEL
    _TWO_DES_WORKER_WHEEL = wheel


def _two_des_point_worker(args):
    t1, t2, t3, sequence, redfield = args
    return _TWO_DES_WORKER_WHEEL.two_des_spectrum_1pt(
        t1, t2, t3, sequence=sequence
    )


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
    def __init__(self, center: tuple[float, float], scale: float, n: int =4, angles: list[float] = None, offset: float = 0):
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
            J = frenkel_coupling(self.center_angle, phi_j, self.angle_step * i + self.offset, scale)
            all_edges.append((0, i + 1, J))
            if NEW_J_CONDITION:
                if J ** 2 > J_SQUARED_THRESHOLD:
                    active_edges.append((0, i + 1))
            else:
                if J < 0:
                    active_edges.append((0, i + 1))
        for i in range(self.n): # between adjacent spokes
            theta_ij = np.arctan2(self.spokes[(i+1)%self.n][1] - self.spokes[i][1], self.spokes[(i+1)%self.n][0] - self.spokes[i][0])
            J = frenkel_coupling(self.spoke_angles[i], self.spoke_angles[(i + 1) % self.n], theta_ij, np.sqrt(2 - 2 * np.cos(2 * np.pi / self.n)) * scale)
            all_edges.append((i + 1, (i + 1) % self.n + 1, J))
            if NEW_J_CONDITION:
                if J ** 2 > J_SQUARED_THRESHOLD:
                    active_edges.append((i + 1, (i + 1) % self.n + 1))
            else: 
                if J < 0:
                    active_edges.append((i + 1, (i + 1) % self.n + 1))
        self.active_edges = active_edges
        self.all_edges = all_edges
        
    def render_wheel(self, ax, draw_dipoles=False, node_size=100):
        '''Render the wheel graph on the given matplotlib axis `ax`. Active edges are colored green, while inactive edges are colored gray.'''
        for (i, j, _) in self.all_edges:
            if (i, j) in self.active_edges:
                color = '#1a9850'  # bright green
                lw = 3
            else:
                color = '#EEEEEE'  # light gray
                lw = 1
            if (i, j)[0] == 0:  # spoke edge
                start = self.center
                end = self.spokes[j - 1]
            else:  # rim edge
                start = self.spokes[i - 1]
                end = self.spokes[j - 1]
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw)
        ax.scatter(*self.center, color='blue', s=node_size, zorder=3)  # center node
        ax.scatter(self.spokes[:, 0], self.spokes[:, 1], color='#AAAAAA', s=node_size, zorder=3)  # spoke nodes
        if draw_dipoles:
            for i in range(self.n):
                angle = self.spoke_angles[i]
                dx = 0.2 * np.cos(angle)
                dy = 0.2 * np.sin(angle)
                ax.arrow(self.spokes[i][0], self.spokes[i][1], dx, dy, head_width=0.05, head_length=0.1, fc='red', ec='red', zorder=4)
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

    def construct_hamiltonian(self, epsilon=150.0):
        '''Construct the Hamiltonian matrix for the wheel graph.
        Values are zero for inactive edges and J for active edges. edge = row i, col j if i and j are connected. 
        We use epsilon = 150 because J = 1 being 100 inverse cm and epsilon being 15000 is somewhat realistic for a molecular system. The diagonal is set to epsilon.
        '''
        matrix = np.zeros((self.n + 1, self.n + 1), dtype='float64')  # initialize Hamiltonian matrix

        np.fill_diagonal(matrix, [epsilon] * (self.n + 1))  # set diagonal to epsilon

        for u, v, J in self.all_edges:
            matrix[u, v] = J
            matrix[v, u] = J

        return matrix

    def compute_eigen(self, epsilon=150.0):
        '''Compute the eigenvalues and eigenvectors of the Hamiltonian matrix for the wheel graph.'''
        H = self.construct_hamiltonian(epsilon)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvalues, eigenvectors

    def two_des_spectrum_1pt(self, t1, t2, t3, sequence='XXXX'):
        '''Compute the 2DES spectrum for a single point in time.'''
        ham = self.construct_hamiltonian()
        ham = np.vstack((np.zeros((1, self.n + 1)), ham))
        ham = np.hstack((np.zeros((self.n + 2, 1)), ham))
        rho_0 = np.zeros((self.n + 2, self.n + 2), dtype=complex)
        rho_0[0, 0] = 1.0
        td_op_x = [0, np.cos(self.center_angle)] + [np.cos(angle) for angle in self.spoke_angles]
        td_op_y = [0, np.sin(self.center_angle)] + [np.sin(angle) for angle in self.spoke_angles]

        td_op_x = np.array(td_op_x, dtype=complex)
        td_op_x.reshape((1, -1))
        td_op_x = np.vstack((td_op_x, np.zeros((self.n + 1, self.n + 2), dtype=complex)))
        td_op_lowering_x = td_op_x
        td_op_raising_x = td_op_x.T

        td_op_y = np.array(td_op_y, dtype=complex)
        td_op_y.reshape((1, -1))
        td_op_y = np.vstack((td_op_y, np.zeros((self.n + 1, self.n + 2), dtype=complex)))
        td_op_lowering_y = td_op_y
        td_op_raising_y = td_op_y.T

        # print('Raising x:', td_op_raising_x)
        # print('Raising y:', td_op_raising_y)
        # print('Lowering x:', td_op_lowering_x)
        # print('Lowering y:', td_op_lowering_y)

        
        def operation_naive(rho, left, raising, x):
            if left:
                if raising:
                    return td_op_raising_x @ rho if x else td_op_raising_y @ rho
                else: 
                    return td_op_lowering_x @ rho if x else td_op_lowering_y @ rho
            else:
                if raising:
                    return rho @ td_op_raising_x if x else rho @ td_op_raising_y
                else: 
                    return rho @ td_op_lowering_x if x else rho @ td_op_lowering_y

        def operation(rho, left, raising, x):
            return operation_naive(rho, left, raising, x) - operation_naive(rho, not left, raising, x)

        U_1 = expm(-1j * t1 * ham)
        U_2 = expm(-1j * t2 * ham)
        U_3 = expm(-1j * t3 * ham)
        pathways = []
        # nonrephasing ESA
        rho_N_ESA = rho_0.copy()
        rho_N_ESA = operation(rho_N_ESA, True, True, sequence[0] == 'X')
        rho_N_ESA = U_1 @ rho_N_ESA @ U_1.conj().T
        rho_N_ESA = operation(rho_N_ESA, False, False, sequence[1] == 'X')
        rho_N_ESA = U_2 @ rho_N_ESA @ U_2.conj().T
        rho_N_ESA = operation(rho_N_ESA, True, True, sequence[2] == 'X')
        rho_N_ESA = U_3 @ rho_N_ESA @ U_3.conj().T
        pathways.append(rho_N_ESA)
        # rephasing ESA
        rho_R_ESA = rho_0.copy()
        rho_R_ESA = operation(rho_R_ESA, False, False, sequence[0] == 'X')
        rho_R_ESA = U_1 @ rho_R_ESA @ U_2.conj().T
        rho_R_ESA = operation(rho_R_ESA, True, True, sequence[1] == 'X')
        rho_R_ESA = U_2 @ rho_R_ESA @ U_2.conj().T
        rho_R_ESA = operation(rho_R_ESA, False, False, sequence[2] == 'X')
        rho_R_ESA = U_3 @ rho_R_ESA @ U_3.conj().T
        pathways.append(rho_R_ESA)
        # nonrephasing GSB
        rho_N_GSB = rho_0.copy()
        rho_N_GSB = operation(rho_N_GSB, True, True, sequence[0] == 'X')
        rho_N_GSB = U_1 @ rho_N_GSB @ U_2.conj().T
        rho_N_GSB = operation(rho_N_GSB, True, False, sequence[1] == 'X')
        rho_N_GSB = U_2 @ rho_N_GSB @ U_2.conj().T
        rho_N_GSB = operation(rho_N_GSB, True, True, sequence[2] == 'X')
        rho_N_GSB = U_3 @ rho_N_GSB @ U_3.conj().T
        pathways.append(rho_N_GSB)
        # rephasing GSB
        rho_R_GSB = rho_0.copy()
        rho_R_GSB = operation(rho_R_GSB, False, False, sequence[0] == 'X')
        rho_R_GSB = U_1 @ rho_R_GSB @ U_2.conj().T
        rho_R_GSB = operation(rho_R_GSB, False, True, sequence[1] == 'X')
        rho_R_GSB = U_2 @ rho_R_GSB @ U_2.conj().T
        rho_R_GSB = operation(rho_R_GSB, False, False, sequence[2] == 'X')
        rho_R_GSB = U_3 @ rho_R_GSB @ U_3.conj().T
        pathways.append(rho_R_GSB)
        # nonresphasing SE
        rho_N_SE = rho_0.copy()
        rho_N_SE = operation(rho_N_SE, True, True, sequence[0] == 'X')
        rho_N_SE = U_1 @ rho_N_SE @ U_2.conj().T
        rho_N_SE = operation(rho_N_SE, False, False, sequence[1] == 'X')
        rho_N_SE = U_2 @ rho_N_SE @ U_2.conj().T
        rho_N_SE = operation(rho_N_SE, True, False, sequence[2] == 'X')
        rho_N_SE = U_3 @ rho_N_SE @ U_3.conj().T
        pathways.append(rho_N_SE)
        # rephasing SE
        rho_R_SE = rho_0.copy()
        rho_R_SE = operation(rho_R_SE, False, False, sequence[0] == 'X')
        rho_R_SE = U_1 @ rho_R_SE @ U_2.conj().T
        rho_R_SE = operation(rho_R_SE, True, True, sequence[1] == 'X')
        rho_R_SE = U_2 @ rho_R_SE @ U_2.conj().T
        rho_R_SE = operation(rho_R_SE, False, True, sequence[2] == 'X')
        rho_R_SE = U_3 @ rho_R_SE @ U_3.conj().T
        pathways.append(rho_R_SE)

        rho_total = sum(pathways)

        if sequence[3] == 'X':
            return np.trace(td_op_x @ rho_total)
        else: 
            return np.trace(td_op_y @ rho_total)


        # rho_S_N_ESA = rho_0.copy()
        # # Pulse 1: raising on ket/left side
        # if sequence[0] == 'X': 
        #     rho_S_N_ESA = td_op_raising_x @ rho_S_N_ESA
        # else:
        #     rho_S_N_ESA = td_op_raising_y @ rho_S_N_ESA

        # # Evolve for t1
        # U = expm(-1j * t1 * ham)
        # rho_S_N_ESA = U @ rho_S_N_ESA @ U.conj().T

        # # Pulse 2: acts on bra/right side
        # if sequence[1] == 'X':
        #     rho_S_N_ESA = td_op_lowering_x @ rho_S_N_ESA
        # else:
        #     td_op = td_op_lowering_y

        # # Evolve for t2
        # U = expm(-1j * t2 * ham)
        # rho_S_N_ESA = U @ rho_S_N_ESA @ U.conj().T

        # # Pulse 3: lowering on ket/left side
        # if sequence[2] == 'X':
        #     rho_S_N_ESA = td_op_lowering_x @ rho_S_N_ESA
        # else: 
        #     rho_S_N_ESA = td_op_lowering_y @ rho_S_N_ESA

        # # Evolve for t3
        # U = expm(-1j * t3 * ham)
        # rho_S_N_ESA = U @ rho_S_N_ESA @ U.conj().T

        # # The signal
        # return np.trace(td_op @ rho_S_N_ESA)

    def generate_t1_states(
        self,
        t1_values,
        sequence='XXXX',
        eta=0.01,
        cutoff=2.0,
        site_coupling=1.0,
        temperature=2.0,
        sec_cutoff=-1.0,
        nsteps=1000000,
    ):
        '''Generate both post-pulse states for every requested ``t1`` value.'''
        try:
            from qutip import Qobj, brmesolve
        except ImportError as exc:
            raise ImportError(
                'generate_t1_states requires the qutip package'
            ) from exc
        if eta < 0:
            raise ValueError('eta must be non-negative')
        if cutoff <= 0:
            raise ValueError('cutoff must be positive')
        if temperature < 0:
            raise ValueError('temperature must be non-negative')
        if nsteps < 1:
            raise ValueError('nsteps must be positive')
        t1_values = np.asarray(t1_values, dtype=float)
        if t1_values.ndim != 1:
            raise ValueError('t1_values must be one-dimensional')
        if np.any(t1_values < 0):
            raise ValueError('t1_values must be non-negative')
        if t1_values.size == 0:
            return np.empty((0, 2, self.n + 2, self.n + 2), dtype=complex)
        if np.any(np.diff(t1_values) < 0):
            raise ValueError('t1_values must be sorted in ascending order')
        solver_times = np.unique(np.concatenate(([0.0], t1_values)))

        ham = self.construct_hamiltonian()
        ham -= 150.0 * np.eye(self.n + 1)
        ham = np.vstack((np.zeros((1, self.n + 1)), ham))
        ham = np.hstack((np.zeros((self.n + 2, 1)), ham))
        ham_qobj = Qobj(ham)

        td_op_x = np.array(
            [0, np.cos(self.center_angle)]
            + [np.cos(angle) for angle in self.spoke_angles],
            dtype=complex,
        )
        td_op_x = np.vstack((
            td_op_x,
            np.zeros((self.n + 1, self.n + 2), dtype=complex),
        ))
        td_op_y = np.array(
            [0, np.sin(self.center_angle)]
            + [np.sin(angle) for angle in self.spoke_angles],
            dtype=complex,
        )
        td_op_y = np.vstack((
            td_op_y,
            np.zeros((self.n + 1, self.n + 2), dtype=complex),
        ))
        lowering = td_op_x if sequence[0] == 'X' else td_op_y
        raising = lowering.T
        rho_0 = np.zeros((self.n + 2, self.n + 2), dtype=complex)
        rho_0[0, 0] = 1.0
        initial_states = (
            raising @ rho_0 - rho_0 @ raising,
            rho_0 @ lowering - lowering @ rho_0,
        )

        redfield_operators = []
        for site in range(1, self.n + 2):
            coupling_operator = np.zeros_like(ham, dtype=complex)
            coupling_operator[site, site] = site_coupling

            def ohmic_spectrum(omega):
                omega = float(omega)
                frequency = abs(omega)
                if frequency == 0.0:
                    return 0.0
                spectral_density = (
                    2.0 * np.pi * eta * frequency
                    * np.exp(-frequency / cutoff)
                )
                if temperature == 0.0:
                    return spectral_density if omega > 0.0 else 0.0
                occupation = 1.0 / np.expm1(frequency / temperature)
                return spectral_density * (
                    occupation + 1.0 if omega > 0.0 else occupation
                )

            redfield_operators.append((Qobj(coupling_operator), ohmic_spectrum))

        def propagate(rho):
            if solver_times[-1] == 0.0:
                return [rho.copy()]
            result = brmesolve(
                ham_qobj,
                Qobj(rho),
                solver_times,
                redfield_operators,
                sec_cutoff=sec_cutoff,
                options={'nsteps': nsteps},
            )
            states_by_time = {
                round(float(time), 12): state.full()
                for time, state in zip(solver_times, result.states)
            }
            return [
                states_by_time[round(float(time), 12)]
                for time in t1_values
            ]

        states_a = propagate(initial_states[0])
        states_b = propagate(initial_states[1])
        return np.asarray(
            list(zip(states_a, states_b)),
            dtype=complex,
        )

    def two_des_spectrum_1row_redfield(
        self,
        state_after_t1,
        t2,
        t3_values,
        sequence='XXXX',
        eta=0.01,
        cutoff=2.0,
        site_coupling=1.0,
        temperature=2.0,
        sec_cutoff=-1.0,
        nsteps=1000000,
    ):
        '''Compute one 2DES row using non-secular Redfield propagation.

        ``state_after_t1`` contains the two post-pulse states for one ``t1``.
        ``t3_values`` contains all detection times for that row.
        Each pathway is propagated through the complete ``t3_values`` grid,
        so the row is returned with one solver pass per pathway rather than
        one solver pass per 2DES point.

        Parameters
        ----------
        eta : float, optional
            Dimensionless Ohmic bath coupling strength. Defaults to ``0.01``.
        cutoff : float, optional
            Ohmic bath cutoff frequency. Defaults to ``2.0``.
        site_coupling : float, optional
            Amplitude of each site's energy-fluctuation operator. Defaults to
            ``1.0``.
        temperature : float, optional
            Bath temperature in the same energy units as the Hamiltonian, with
            Boltzmann's constant set to one. Defaults to ``2.0``.
            2.0 is because the energy scale of the Hamiltonian is ~100 inverse cm,
            and 2.0 corresponds to 300K, which is a reasonable temperature for
            a typical 2DES reading.
        sec_cutoff : float, optional
            QuTiP Redfield secular cutoff. A negative value disables the
            secular approximation; the default is ``-1.0``.
        nsteps : int, optional
            Maximum number of internal integration steps for QuTiP's solver
            per requested output interval. Defaults to ``1000000``.
        '''
        try:
            from qutip import Qobj, brmesolve
        except ImportError as exc:
            raise ImportError(
                'two_des_spectrum_1row_redfield requires the qutip package'
            ) from exc
        if eta < 0:
            raise ValueError('eta must be non-negative')
        if cutoff <= 0:
            raise ValueError('cutoff must be positive')
        if temperature < 0:
            raise ValueError('temperature must be non-negative')
        if nsteps < 1:
            raise ValueError('nsteps must be positive')
        state_after_t1 = np.asarray(state_after_t1, dtype=complex)
        if state_after_t1.shape != (2, self.n + 2, self.n + 2):
            raise ValueError(
                'state_after_t1 must have shape (2, n + 2, n + 2)'
            )
        t3_values = np.asarray(t3_values, dtype=float)
        if t3_values.ndim != 1:
            raise ValueError('t3_values must be one-dimensional')
        if np.any(t3_values < 0):
            raise ValueError('t3_values must be non-negative')
        solver_times = np.unique(np.concatenate(([0.0], t3_values)))

        ham = self.construct_hamiltonian()

        omega_ref = 150.0 # Shift into rotating frame so we don't have to make dt absolutely tiny. 
        ham -= omega_ref * np.eye(self.n + 1) # All this does is reduce frequencies so we don't hit the Nyquist limit with our dt of 0.1. 

        ham = np.vstack((np.zeros((1, self.n + 1)), ham))
        ham = np.hstack((np.zeros((self.n + 2, 1)), ham))
        ham_qobj = Qobj(ham)

        td_op_x = [0, np.cos(self.center_angle)] + [
            np.cos(angle) for angle in self.spoke_angles
        ]
        td_op_y = [0, np.sin(self.center_angle)] + [
            np.sin(angle) for angle in self.spoke_angles
        ]

        td_op_x = np.array(td_op_x, dtype=complex)
        td_op_x = np.vstack((
            td_op_x,
            np.zeros((self.n + 1, self.n + 2), dtype=complex),
        ))
        td_op_lowering_x = td_op_x
        td_op_raising_x = td_op_x.T

        td_op_y = np.array(td_op_y, dtype=complex)
        td_op_y = np.vstack((
            td_op_y,
            np.zeros((self.n + 1, self.n + 2), dtype=complex),
        ))
        td_op_lowering_y = td_op_y
        td_op_raising_y = td_op_y.T

        # Site-energy fluctuations do not couple the ground state.  A
        # non-secular Redfield treatment is requested with sec_cutoff=-1.
        redfield_operators = []
        for site in range(1, self.n + 2):
            coupling_operator = np.zeros_like(ham, dtype=complex)
            coupling_operator[site, site] = site_coupling
            coupling_operator = Qobj(coupling_operator)

            def ohmic_spectrum(omega):
                omega = float(omega)
                frequency = abs(omega)
                if frequency == 0.0:
                    return 0.0
                spectral_density = (
                    2.0 * np.pi * eta * frequency
                    * np.exp(-frequency / cutoff)
                )
                if temperature == 0.0:
                    return spectral_density if omega > 0.0 else 0.0
                occupation = np.expm1(frequency / temperature)
                occupation = 1.0 / occupation
                return spectral_density * (
                    occupation + 1.0 if omega > 0.0 else occupation
                )

            redfield_operators.append((coupling_operator, ohmic_spectrum))

        def propagate(rho, times):
            times = np.asarray(times, dtype=float)
            if times[-1] == times[0]:
                return [rho]
            result = brmesolve(
                ham_qobj,
                Qobj(rho),
                times,
                redfield_operators,
                sec_cutoff=sec_cutoff,
                options={'nsteps': nsteps},
            )
            return [state.full() for state in result.states]

        def operation_naive(rho, left, raising, x):
            if left:
                if raising:
                    return td_op_raising_x @ rho if x else td_op_raising_y @ rho
                return td_op_lowering_x @ rho if x else td_op_lowering_y @ rho
            if raising:
                return rho @ td_op_raising_x if x else rho @ td_op_raising_y
            return rho @ td_op_lowering_x if x else rho @ td_op_lowering_y

        def operation(rho, left, raising, x):
            return operation_naive(rho, left, raising, x) - operation_naive(
                rho, not left, raising, x
            )

        signal_operator = td_op_x if sequence[3] == 'X' else td_op_y
        pathway_specs = (
            (state_after_t1[0], False, False, True, True),
            (state_after_t1[1], True, True, False, False),
            (state_after_t1[0], True, False, True, True),
            (state_after_t1[1], False, True, False, False),
            (state_after_t1[0], False, False, True, False),
            (state_after_t1[1], True, True, False, True),
        )
        pathways = []
        for rho, left_t2, raising_t2, left_t3, raising_t3 in pathway_specs:
            rho = operation(rho, left_t2, raising_t2, sequence[1] == 'X')
            rho = propagate(rho, [0.0, t2])[-1]
            rho = operation(rho, left_t3, raising_t3, sequence[2] == 'X')
            states = propagate(rho, solver_times)
            states_by_time = {
                round(float(time), 12): state
                for time, state in zip(solver_times, states)
            }
            pathways.append([
                states_by_time[round(float(time), 12)]
                for time in t3_values
            ])

        rho_totals = [
            sum(pathway[index] for pathway in pathways)
            for index in range(t3_values.size)
        ]
        return np.asarray([
            np.trace(signal_operator @ rho_total)
            for rho_total in rho_totals
        ])

    def two_des_spectrum(
        self, t2, sequence='XXXX', redfield=False, n_processes=None
    ):
        '''Compute a 2DES spectrum, evaluating time points in parallel.

        ``n_processes`` controls the process-pool size for both calculation
        modes. Redfield rows are evaluated concurrently after all ``t1``
        states have been generated.
        '''
        dt1 = 0.1
        dt3 = 0.1
        t1 = np.arange(0, 25, dt1)
        t3 = np.arange(0, 25, dt3)
        if n_processes is not None and n_processes < 1:
            raise ValueError('n_processes must be positive')
        if redfield:
            t1_states = self.generate_t1_states(t1, sequence=sequence)
            # We removed concurrency because it was causing issues with the QuTiP solver. The code below is the original implementation that used ThreadPoolExecutor to compute rows in parallel, but it has been commented out.
            # row_args = [
            #     (state_after_t1, t2, t3, sequence)
            #     for state_after_t1 in t1_states
            # ]
            # if n_processes is None:
            #     n_processes = min(cpu_count(), len(row_args))
            # n_processes = min(n_processes, len(row_args))
            # with ThreadPoolExecutor(max_workers=n_processes) as executor:
            #     rows = executor.map(
            #         lambda args: self.two_des_spectrum_1row_redfield(*args),
            #         row_args,
            #     )
            #     S = np.asarray(list(rows), dtype=complex).T
            S = []
            for state_after_t1 in t1_states:
                row = self.two_des_spectrum_1row_redfield(
                    state_after_t1, t2, t3, sequence=sequence
                )
                S.append(row)
            S = np.asarray(S, dtype=complex).T
        else:
            T1, T3 = np.meshgrid(t1, t3)
            tasks = [
                (T1[i, j], t2, T3[i, j], sequence, redfield)
                for i in range(T1.shape[0])
                for j in range(T1.shape[1])
            ]
            if n_processes is None:
                n_processes = min(cpu_count(), len(tasks))
            if n_processes < 1:
                raise ValueError('n_processes must be positive')
            n_processes = min(n_processes, len(tasks))
            print("Using {} processes for 2DES spectrum computation".format(n_processes))
            chunksize = max(1, len(tasks) // (n_processes * 20))
            with Pool(
                processes=n_processes,
                initializer=_init_two_des_worker,
                initargs=(self,),
            ) as pool:
                values = pool.map(
                    _two_des_point_worker, tasks, chunksize=chunksize
                )
            S = np.asarray(values, dtype=complex).reshape(T1.shape)

        # 2D Fourier transform
        S_fft = np.fft.fftshift(
            np.fft.fft2(S, axes=(0, 1)),
            axes=(0, 1)
        )

        # np.fft.fftfreq gives ordinary frequency f (cycles / time).
        # Multiply by 2*pi to obtain angular frequency omega.
        omega1 = 2 * np.pi * np.fft.fftshift(
            np.fft.fftfreq(len(t1), d=dt1)
        )

        omega3 = 2 * np.pi * np.fft.fftshift(
            np.fft.fftfreq(len(t3), d=dt3)
        )

        # Restore absolute optical frequencies
        omega_ref = 150.0

        omega1 -= omega_ref
        omega3 += omega_ref

        return omega1, omega3, S_fft

    def plot_2des(self, ax, t2=1.0, zoom_factor=1.0, log=False, redfield=False, n_processes=None):
        '''Plot the 2DES spectrum.

        The Hamiltonian and time variables use units where ``K_s = 1`` and
        ``hbar = 1``.  Since ``K_s`` represents ``100 cm^-1`` in this model,
        the angular frequencies returned by :meth:`two_des_spectrum` and
        plotted here are normalized by ``K_s``: a plotted value of
        ``omega = 1`` corresponds to ``100 cm^-1`` (and ``omega = 0.01``
        corresponds to ``1 cm^-1``).  The factor of ``2*pi`` in the FFT
        converts the FFT's cycles-per-time frequencies to angular
        frequencies, consistent with the Hamiltonian evolution convention.
        '''
        omega1, omega3, S = self.two_des_spectrum(t2, redfield=redfield, n_processes=n_processes)

        zoom_factor = float(zoom_factor)
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be positive")
        zoom_factor = min(zoom_factor, 1.0)

        n1 = len(omega1)
        n3 = len(omega3)
        keep_n1 = max(1, int(round(n1 * zoom_factor)))
        keep_n3 = max(1, int(round(n3 * zoom_factor)))

        start1 = (n1 - keep_n1) // 2
        end1 = start1 + keep_n1
        start3 = (n3 - keep_n3) // 2
        end3 = start3 + keep_n3

        omega1 = omega1[start1:end1]
        omega3 = omega3[start3:end3]
        S = S[start3:end3, start1:end1]

        if log:
            S = np.log(np.abs(S) + 1e-10)
        else:
            S = np.abs(S)

        im = ax.imshow(
            S,
            extent=(omega1.min(), omega1.max(), omega3.min(), omega3.max()),
            origin='lower',
            aspect='auto',
            cmap='viridis',
        )
        ax.set_xlabel(r'$\omega_1/K_s$')
        ax.set_ylabel(r'$\omega_3/K_s$')
        ax.set_title(f'2DES Spectrum at $t_2={t2}$')
        if not log:
            plt.colorbar(im, ax=ax, label='Signal Intensity')
        else: 
            plt.colorbar(im, ax=ax, label='Log Signal Intensity')

    def two_des_anisotropy(self, t2, redfield=False, n_processes=None):
        omega1, omega3, S_xxxx = self.two_des_spectrum(
            t2, sequence='XXXX', redfield=redfield, n_processes=n_processes
        )
        _, _, S_xyxy = self.two_des_spectrum(
            t2, sequence='XYXY', redfield=redfield
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            anisotropy = (S_xxxx - S_xyxy) / (S_xxxx + 2 * S_xyxy)
        return omega1, omega3, anisotropy

    def plot_2des_anisotropy(
        self, ax, t2=1.0, zoom_factor=1.0, redfield=False
    ):
        omega1, omega3, anisotropy = self.two_des_anisotropy(
            t2, redfield=redfield
        )

        zoom_factor = float(zoom_factor)
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be positive")
        zoom_factor = min(zoom_factor, 1.0)

        n1 = len(omega1)
        n3 = len(omega3)
        keep_n1 = max(1, int(round(n1 * zoom_factor)))
        keep_n3 = max(1, int(round(n3 * zoom_factor)))

        start1 = (n1 - keep_n1) // 2
        end1 = start1 + keep_n1
        start3 = (n3 - keep_n3) // 2
        end3 = start3 + keep_n3

        omega1 = omega1[start1:end1]
        omega3 = omega3[start3:end3]

        anisotropy = np.abs(anisotropy)
        anisotropy = anisotropy[start3:end3, start1:end1]

        im = ax.imshow(
            anisotropy,
            extent=(omega1.min(), omega1.max(), omega3.min(), omega3.max()),
            origin='lower',
            aspect='auto',
            vmin=-0.5,
            vmax=0.5
        )
        ax.set_xlabel(r'$\omega_1/K_s$')
        ax.set_ylabel(r'$\omega_3/K_s$')
        ax.set_title(f'2DES Anisotropy at $t_2={t2}$')
        plt.colorbar(im, ax=ax, label='Anisotropy')

    def plot_2des_anisotopy(
        self, ax, t2=1.0, zoom_factor=1.0, redfield=False
    ):
        '''Backward-compatible alias for the correctly spelled method.'''
        return self.plot_2des_anisotropy(
            ax, t2=t2, zoom_factor=zoom_factor, redfield=redfield
        )

    def linear_absorption_1pt(self, t, x=True):
        ham = self.construct_hamiltonian()
        ham = np.vstack((np.zeros((1, self.n + 1)), ham))
        ham = np.hstack((np.zeros((self.n + 2, 1)), ham))
        rho = np.zeros((self.n + 2, self.n + 2), dtype=complex)
        rho[0, 0] = 1.0
        if x:
            td_op = [0, np.cos(self.center_angle)] + [np.cos(angle) for angle in self.spoke_angles]
        else:
            td_op = [0, np.sin(self.center_angle)] + [np.sin(angle) for angle in self.spoke_angles]
        td_op = np.array(td_op, dtype=complex)
        td_op.reshape((1, -1))
        td_op = np.vstack((td_op, np.zeros((self.n + 1, self.n + 2), dtype=complex)))
        td_op += td_op.T  # make it Hermitian
        # Pulse
        rho = -1j * (td_op @ rho - rho @ td_op)
        # Time evolve
        U = expm(-1j * ham * t)
        rho = U @ rho @ U.conj().T

        return np.trace(td_op @ rho)

    def linear_absorption_spectrum(self, x=True, redfield=False):
        dt = 0.1
        t = np.arange(0, 30, dt)
        S = np.zeros_like(t, dtype=complex)
        if redfield:
            try:
                from qutip import Qobj, brmesolve
            except ImportError as exc:
                raise ImportError(
                    'linear_absorption_spectrum requires the qutip package '
                    'when redfield=True'
                ) from exc

            eta = 0.01
            cutoff = 2.0
            site_coupling = 1.0
            temperature = 2.0
            sec_cutoff = -1.0
            nsteps = 1000000
            ham = self.construct_hamiltonian()

            # Rotating frame
            omega_ref = 150.0
            ham -= omega_ref * np.eye(self.n + 1)

            ham = np.vstack((
                np.zeros((1, self.n + 1)), ham
            ))
            ham = np.hstack((
                np.zeros((self.n + 2, 1)), ham
            ))
            ham_qobj = Qobj(ham)

            # Transition dipoles for the selected polarization
            dipoles = np.array(
                [np.cos(self.center_angle)]
                + [np.cos(angle) for angle in self.spoke_angles]
                if x else
                [np.sin(self.center_angle)]
                + [np.sin(angle) for angle in self.spoke_angles],
                dtype=complex,
            )

            # Raising and lowering operators
            raising = np.zeros(
                (self.n + 2, self.n + 2), dtype=complex
            )
            raising[1:, 0] = dipoles
            lowering = raising.conj().T

            rho_0 = np.zeros(
                (self.n + 2, self.n + 2), dtype=complex
            )
            rho_0[0, 0] = 1.0

            # Only the absorption coherence
            rho_after_pulse = raising @ rho_0

            # Redfield bath operators
            redfield_operators = []
            for site in range(1, self.n + 2):
                coupling_operator = np.zeros_like(
                    ham, dtype=complex
                )
                coupling_operator[site, site] = site_coupling

                def ohmic_spectrum(omega):
                    omega = float(omega)
                    frequency = abs(omega)

                    if frequency == 0.0:
                        return 0.0

                    spectral_density = (
                        2.0 * np.pi * eta * frequency
                        * np.exp(-frequency / cutoff)
                    )

                    if temperature == 0.0:
                        return (
                            spectral_density
                            if omega > 0.0 else 0.0
                        )

                    occupation = (
                        1.0 / np.expm1(frequency / temperature)
                    )

                    return spectral_density * (
                        occupation + 1.0
                        if omega > 0.0 else occupation
                    )

                redfield_operators.append(
                    (Qobj(coupling_operator), ohmic_spectrum)
                )

            # Propagate once and collect all snapshots
            solver_times = np.append(t, 30.0)

            result = brmesolve(
                ham_qobj,
                Qobj(rho_after_pulse),
                solver_times,
                redfield_operators,
                sec_cutoff=sec_cutoff,
                options={'nsteps': nsteps},
            )

            # Dipole correlation function
            S = np.asarray([
                np.trace(lowering @ state.full())
                for state in result.states[:t.size]
            ])

        else:
            # Original non-Redfield calculation
            for i in range(t.shape[0]):
                S[i] = self.linear_absorption_1pt(t[i], x)

        # Suppress finite-time truncation artifacts
        gamma_window = 0.05
        S = S * np.exp(-gamma_window * t)

        # Fourier transform with positive excitation-frequency convention
        padding_factor = 16
        n_fft = len(t) * padding_factor

        S_fft = dt * np.fft.fftshift(
            np.fft.fft(np.conj(S), n=n_fft)
        )

        omega = 2 * np.pi * np.fft.fftshift(
            np.fft.fftfreq(n_fft, d=dt)
        )

        # Restore absolute optical frequencies for Redfield mode
        if redfield:
            omega += 150.0

        return omega, S_fft

    def plot_linear_absorption(
        self, ax, x=True, zoom_factor=1.0, redfield=False
    ):
        omega, S = self.linear_absorption_spectrum(x, redfield=redfield)

        zoom_factor = float(zoom_factor)
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be positive")
        zoom_factor = min(zoom_factor, 1.0)

        n = len(omega)
        keep_n = max(1, int(round(n * zoom_factor)))
        start = (n - keep_n) // 2
        end = start + keep_n

        omega = omega[start:end]
        S = S[start:end]

        absorption = np.real(S)

        ax.plot(omega, absorption, color='tab:blue', lw=1.6)
        ax.fill_between(
            omega, 0, absorption,
            color='tab:blue', alpha=0.25
        )
        ax.set_xlabel(r'$\omega/K_s$', fontsize=9)
        ax.set_ylabel('Linear absorption', fontsize=9)
        ax.set_title('Linear Absorption Spectrum', fontsize=9)

class SquareWheelGrid:
    def __init__(self, rows, cols, separation=3, scale=1, angles=None, defects = []):
        '''Class representing a grid of wheels with n=4 spokes. Each wheel is represented by an instance of the `Wheel` class.'''
        self.rows = rows
        self.cols = cols
        self.separation = separation
        self.wheels = []
        defect_locations = [i[0] for i in defects]
        for i in range(rows):
            row = []
            for j in range(cols):
                center = (j * separation, i * separation)
                if (i, j) in defect_locations:
                    wheel = Wheel(center, scale, 4, angles=defects[defect_locations.index((i,j))][1])
                else:
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
                J = frenkel_coupling(phi1, phi2, theta_ij, separation - 2 * scale)
                if NEW_J_CONDITION: 
                    if J ** 2 > J_SQUARED_THRESHOLD:
                        self.active_btwn_wheel_edges.append(((i, j, 1), (i, j + 1, 3)))  # connect spoke 1 of wheel (i, j) to spoke 3 of wheel (i, j + 1)
                        self.horiz_active = True
                else:
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
                J = frenkel_coupling(phi1, phi2, theta_ij, separation - 2 * scale)
                if NEW_J_CONDITION:
                    if J ** 2 > J_SQUARED_THRESHOLD:
                        self.active_btwn_wheel_edges.append(((i, j, 2), (i + 1, j, 4)))  # connect spoke 2 of wheel (i, j) to spoke 4 of wheel (i + 1, j)
                        self.vert_active = True
                else:
                    if J < 0:
                        self.active_btwn_wheel_edges.append(((i, j, 2), (i + 1, j, 4)))  # connect spoke 2 of wheel (i, j) to spoke 4 of wheel (i + 1, j)
                        self.vert_active = True
                btwn_wheel_connections.append(((i, j, 2), (i + 1, j, 4), J))  # connect spoke 2 of wheel (i, j) to spoke 4 of wheel (i + 1, j)
        self.btwn_wheel_connections = btwn_wheel_connections

    def render_grid(self, ax, draw_dipoles=False, node_size=100, linewidth=(1,2)):
        '''Render the grid of wheels on the given matplotlib axis `ax`.'''
        for row in self.wheels:
            for wheel in row:
                wheel.render_wheel(ax, draw_dipoles=draw_dipoles, node_size=node_size)
        for ((row1, col1, spoke1), (row2, col2, spoke2), _) in self.btwn_wheel_connections:
            wheel1 = self.wheels[row1][col1]
            wheel2 = self.wheels[row2][col2]
            start = wheel1.spokes[spoke1 - 1]
            end = wheel2.spokes[spoke2 - 1]
            if ((row1, col1, spoke1), (row2, col2, spoke2)) in self.active_btwn_wheel_edges:
                color = '#1a9850'  # bright green
                lw = linewidth[1]
            else:
                color = '#CCCCCC'  # light gray
                lw = linewidth[0]
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw)
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
    
    def euler_characteristic_density(self):
        e = sum(1 for i in f'{self.binary_representation():b}' if i == '1')
        v = 5
        t = self.num_triangles()
        return v - e + t

    def b1_matrix(self):
        mat = np.zeros((0,0))
        for row in self.wheels:
            for wheel in row:
                mat = np.vstack((mat, np.zeros((5, mat.shape[1]))))
                mat = np.hstack((mat, np.zeros((mat.shape[0], edge_count:=len(wheel.active_edges)))))
                for i, (u, v) in enumerate(wheel.active_edges):
                    u_1, v_1 = min(u, v), max(u, v)
                    mat[u_1 - 5, i - edge_count] = -1
                    mat[v_1 - 5, i - edge_count] = 1
        for i, ((r1, c1, n1), (r2, c2, n2)) in enumerate(self.active_btwn_wheel_edges):
            mat = np.hstack((mat, np.zeros((mat.shape[0], 1))))
            n1_idx = 5 * (r1 * self.cols + c1) + n1
            n2_idx = 5 * (r2 * self.cols + c2) + n2
            n1_idx, n2_idx = min(n1_idx, n2_idx), max(n1_idx, n2_idx)
            mat[n1_idx, -1] = -1
            mat[n2_idx, -1] = 1
        return mat

    def b2_matrix(self):
        mat = np.zeros((len(self.active_btwn_wheel_edges) + sum([sum([len(wheel.active_edges) for wheel in row]) for row in self.wheels]), self.rows * self.cols * self.num_triangles()))
        
        if self.num_triangles() == 0:
            return mat
        #note: edges count up like: wheel1 edges, wheel2, edges, etc... interwheel edges
        #note: wheel edges are counted as center-spoke then rim. so CE CN CW CS NE NW SW SE
        for row in range(self.rows):
            for col in range(self.cols):
                wheel = self.wheels[row][col]
                t = 0
                for i in range(4):
                    if (i+1, (i+1) % 4 + 1) in wheel.active_edges and (0, i+1) in wheel.active_edges and (0, (i+1)%4 + 1) in wheel.active_edges:
                        #1st and 2nd edge are center-spoke, 3rd edge is always the rim one. 
                        a = wheel.active_edges.index((0, i+1)) #amongst the edges, find the index of this one. 
                        b = wheel.active_edges.index((0, (i+1)%4 + 1))
                        c = wheel.active_edges.index((i+1, (i+1) % 4 + 1))
                        a, b, c = sorted([a, b, c]) # put them in order lol
                        mat_row = 5 * (row * self.cols + col)
                        mat_col = self.num_triangles() * (row * self.cols + col) + t
                        mat[mat_row + a, mat_col] = 1
                        mat[mat_row + b, mat_col] = -1
                        mat[mat_row + c, mat_col] = 1
                        t += 1 # so we write in the next spot next time
        return mat

    def l1_laplacian(self):
        b1 = self.b1_matrix()
        b2 = self.b2_matrix()
        return b1.T @ b1 + b2 @ b2.T
        
    def l1_eig(self):
        #returns l1 eigenvalues and eigenvectors
        return eig(self.l1_laplacian())
    
    def l1_eig_dist(self):
        vals, _ = self.l1_eig()
        vals = np.real_if_close(vals)
        vals = np.sort(vals)
        
        values, occurences = [float(vals[0])], [1]
        for val in vals:
            if np.isclose(val, values[-1], rtol=1e-8, atol=1e-10):
                occurences[-1] += 1
            else:
                values.append(float(val))
                occurences.append(1)

        return values, occurences



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
    val = grid.euler_characteristic_density()
    return val

def fig_4_phase_diagram(save=True, show=False, interactive=False, preprocessed=False):
    '''Using the staggered angling, plot 4 figures showing a phase diagram of beta 1 for a 3x3 grid of wheels as a function of the NS angles and EW angles. The 4 figures differ only by center angle.'''
    fig, axes = plt.subplots(2, 2, figsize=(25, 10))
    theta_c_degrees = [0, 10, 45, 80]
    ns_angle_degrees = np.linspace(0, 360, 500)
    ew_angle_degrees = np.linspace(0, 360, 500)
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
    #print("Generating figure 4 phase diagram...")
    #fig_4_phase_diagram(save=True, show=True, interactive=True, preprocessed=False)
    # print("Generating figure 5 phase diagram...")
    # fig_5_phase_diagram(save=True)
    grid = SquareWheelGrid(3, 3, angles = [45 * DEG] + [30 * DEG, 60 * DEG] * 2)
    # ax = plt.subplot()
    # grid.render_grid(ax)
    # plt.show()
    # print(grid.b1_matrix())
    # print(grid.b2_matrix())
    print(grid.l1_laplacian())
    print(grid.l1_eig())
