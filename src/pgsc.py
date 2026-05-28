"""
src/pgsc.py — Polarization-Gated Simplicial Complex (PGSC) builder.

An edge (i, j) exists in the excitonic network iff the Frenkel coupling
J_ij > J_cut (default J_cut = 0, the positive-coupling-only convention).
The resulting graph is a 1-dimensional simplicial complex.  Its topology is
characterised by the Betti numbers:

    β₀  =  number of connected components  (computed via union-find)
    β₁  =  number of independent cycles    (Euler: β₁ = |E| − |V| + β₀)

Frenkel coupling (2-D, point-dipole):

    J(φᵢ, φⱼ, θᵢⱼ) = (μ²/r³)[cos(φᵢ − φⱼ) − 3 cos(φᵢ − θᵢⱼ) cos(φⱼ − θᵢⱼ)]

where θᵢⱼ is the angle of the i→j internuclear axis and r is the distance.

Public API
----------
frenkel_coupling(phi_i, phi_j, theta_ij, r_ij, mu=1.0)
build_pgsc(positions, orientations, r_cut=None, J_cut=0.0, mu=1.0)
w4_positions(d=1.0)
w4_pgsc(theta_C, theta_P, K_s=1.0, J_cut=0.0)
w4_betti(theta_C, theta_P, K_s=1.0)
w4_2x2_positions(d=1.0)
w4_2x2_pgsc(theta_C, theta_alpha, theta_beta, K_s=1.0, J_cut=0.0,
             r_cut=None, d=1.0)
w4_2x2_betti(theta_C, theta_alpha, theta_beta, K_s=1.0, tier=1, d=1.0)

Node index convention for the W₄ single tile (matches src/two_des.py):
    C=0, N=1, E=2, S=3, W=4

Node index convention for the 2×2 assembly (tile × 5 + local):
    Tile 0 = NW (center at (−3d/2, +3d/2))
    Tile 1 = NE (center at (+3d/2, +3d/2))
    Tile 2 = SW (center at (−3d/2, −3d/2))
    Tile 3 = SE (center at (+3d/2, −3d/2))
    Within each tile: [C=0, N=1, E=2, S=3, W=4]
    Global index = tile_idx * 5 + local_idx
"""

import numpy as np


# ---------------------------------------------------------------------------
# Frenkel coupling (generic)
# ---------------------------------------------------------------------------

def frenkel_coupling(phi_i, phi_j, theta_ij, r_ij, mu=1.0):
    """Compute the Frenkel coupling J(φᵢ, φⱼ, θᵢⱼ).

    J = (μ²/r³)[cos(φᵢ − φⱼ) − 3 cos(φᵢ − θᵢⱼ) cos(φⱼ − θᵢⱼ)]

    Parameters
    ----------
    phi_i, phi_j : float  (radians)  Dipole orientation angles.
    theta_ij     : float  (radians)  Angle of the i→j axis from +x.
    r_ij         : float             Internuclear distance.
    mu           : float             Dipole magnitude.

    Returns
    -------
    J : float
    """
    return (mu ** 2 / r_ij ** 3) * (
        np.cos(phi_i - phi_j)
        - 3.0 * np.cos(phi_i - theta_ij) * np.cos(phi_j - theta_ij)
    )


# ---------------------------------------------------------------------------
# Connected-components counter (union-find with path compression)
# ---------------------------------------------------------------------------

def _count_components(n_vertices, edges):
    """Return β₀ (number of connected components) via union-find."""
    parent = list(range(n_vertices))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in edges:
        union(i, j)

    return sum(1 for x in range(n_vertices) if find(x) == x)


# ---------------------------------------------------------------------------
# Generic PGSC builder
# ---------------------------------------------------------------------------

def build_pgsc(positions, orientations, r_cut=None, J_cut=0.0, mu=1.0):
    """Build the Polarization-Gated Simplicial Complex.

    Edge (i, j) is included iff r_ij ≤ r_cut (or r_cut is None) AND J_ij > J_cut.

    Parameters
    ----------
    positions    : (N, 2) array-like  Chromophore (x, y) positions.
    orientations : (N,) array-like    Dipole orientation angles (radians).
    r_cut        : float or None      Maximum internuclear distance.
    J_cut        : float              Coupling threshold (edges need J > J_cut).
    mu           : float              Dipole magnitude (sets energy scale μ²/r³).

    Returns
    -------
    edges  : list of (int, int)  Active edge pairs (i < j).
    beta0  : int                 Number of connected components.
    beta1  : int                 Number of independent cycles (β₁ = |E|−|V|+β₀).
    """
    positions = np.asarray(positions, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    N = len(positions)

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            r_ij = np.hypot(dx, dy)
            if r_cut is not None and r_ij > r_cut + 1e-12:
                continue
            theta_ij = np.arctan2(dy, dx)
            J = frenkel_coupling(orientations[i], orientations[j],
                                 theta_ij, r_ij, mu)
            if J < J_cut:
                edges.append((i, j))

    beta0 = _count_components(N, edges)
    beta1 = len(edges) - N + beta0
    return edges, beta0, beta1


# ---------------------------------------------------------------------------
# W₄ single tile — positions and analytic PGSC
# ---------------------------------------------------------------------------

def w4_positions(d=1.0):
    """Return the (5, 2) node positions for the W₄ wheel graph.

    Node order: C=0, N=1, E=2, S=3, W=4  (matching src/two_des.py).

    Parameters
    ----------
    d : float  Spoke length.
    """
    return np.array([
        [ 0.0,  0.0],   # C
        [ 0.0,  d  ],   # N
        [ d,    0.0],   # E
        [ 0.0, -d  ],   # S
        [-d,    0.0],   # W
    ])


def w4_pgsc(theta_C, theta_P, K_s=1.0, J_cut=0.0):
    """Compute the PGSC for a W₄ single tile using analytic coupling formulas.

    Uses the exact expressions from THEORY.md §1.3 rather than the generic
    Frenkel function, so K_s is the explicit energy scale.

    Parameters
    ----------
    theta_C, theta_P : float  Orientation angles (radians).
    K_s              : float  Spoke coupling scale (μ²/d³).
    J_cut            : float  Edge threshold (default 0.0).

    Returns
    -------
    edges  : list of (int, int)
    beta0  : int
    beta1  : int
    """
    K_r    = K_s / (2.0 * np.sqrt(2.0))
    J_NS   = K_s * (np.cos(theta_C) * np.cos(theta_P)
                    - 2.0 * np.sin(theta_C) * np.sin(theta_P))
    J_WE   = K_s * (np.sin(theta_C) * np.sin(theta_P)
                    - 2.0 * np.cos(theta_C) * np.cos(theta_P))
    J_diag = K_r * (-0.5 + 1.5 * np.sin(2.0 * theta_P))
    J_anti = K_r * (-0.5 - 1.5 * np.sin(2.0 * theta_P))

    # All 8 edges with their couplings
    # Spokes: C-N (0-1), C-S (0-3), C-E (0-2), C-W (0-4)
    # Rim:    N-E (1-2, diag), S-W (3-4, diag), E-S (2-3, anti), W-N (4-1, anti)
    candidate = [
        (0, 1, J_NS),    # C–N
        (0, 3, J_NS),    # C–S
        (0, 2, J_WE),    # C–E
        (0, 4, J_WE),    # C–W
        (1, 2, J_diag),  # N–E
        (3, 4, J_diag),  # S–W
        (2, 3, J_anti),  # E–S
        (1, 4, J_anti),  # W–N
    ]
    edges = [(i, j) for i, j, J in candidate if J < J_cut]
    beta0 = _count_components(5, edges)
    beta1 = len(edges) - 5 + beta0
    return edges, beta0, beta1


def w4_betti(theta_C, theta_P, K_s=1.0):
    """Return (β₀, β₁) for the W₄ single tile."""
    _, beta0, beta1 = w4_pgsc(theta_C, theta_P, K_s)
    return beta0, beta1


# ---------------------------------------------------------------------------
# 2×2 staggered tiling — positions and numeric PGSC
# ---------------------------------------------------------------------------

def w4_2x2_positions(d=1.0):
    """Return the (20, 2) node positions for the 2×2 W₄ assembly.

    Tile centers are placed at separation 3d so that the nearest inter-tile
    rim-to-rim distance equals d (same scale as intra-tile spokes).

    Layout:
        Tile 0 (NW): center (−3d/2, +3d/2)
        Tile 1 (NE): center (+3d/2, +3d/2)
        Tile 2 (SW): center (−3d/2, −3d/2)
        Tile 3 (SE): center (+3d/2, −3d/2)

    Global index = tile_idx * 5 + local_idx,  local order [C,N,E,S,W].
    """
    tile_centers = np.array([
        [-1.5 * d,  1.5 * d],   # NW (tile 0)
        [ 1.5 * d,  1.5 * d],   # NE (tile 1)
        [-1.5 * d, -1.5 * d],   # SW (tile 2)
        [ 1.5 * d, -1.5 * d],   # SE (tile 3)
    ])
    local = w4_positions(d)     # (5, 2) offsets from tile center
    return np.vstack([tc + local for tc in tile_centers])


def w4_2x2_pgsc(theta_C, theta_alpha, theta_beta,
                K_s=1.0, J_cut=0.0, r_cut=None, d=1.0):
    """Compute the PGSC for the 2×2 staggered W₄ assembly.

    Staggered-texture orientation assignment (all four tiles share the same
    angular parameters):
        Center (C) nodes  →  theta_C
        N and S rim nodes →  theta_alpha
        E and W rim nodes →  theta_beta

    The generic build_pgsc is used with dipole magnitude
    μ = √(K_s · d³) so that the spoke coupling scale equals K_s.

    Parameters
    ----------
    theta_C, theta_alpha, theta_beta : float  (radians)
    K_s    : float  Spoke coupling scale.
    J_cut  : float  Edge threshold.
    r_cut  : float or None
        Distance cutoff for edge inclusion.
        None → Tier-1 default: d·√2·1.001, which includes all intra-tile
        edges (spokes at d, rim at d√2) and the four nearest inter-tile
        edges at d.  For Tier-2 (Group A) set r_cut = d·√5·1.001.
    d      : float  Spoke length.

    Returns
    -------
    edges  : list of (int, int)
    beta0  : int
    beta1  : int
    """
    positions = w4_2x2_positions(d)

    # Orientation array: local index 0=C, 1=N, 2=E, 3=S, 4=W
    _local_ori = np.array([theta_C, theta_alpha, theta_beta,
                            theta_alpha, theta_beta])
    orientations = np.tile(_local_ori, 4)   # (20,)

    # Tier-1 default: include spokes (d) + rim (d√2) + nearest inter-tile (d)
    if r_cut is None:
        r_cut = d * np.sqrt(2.0) * 1.001

    # μ² = K_s · d³  →  (μ²/r³)|_{r=d} = K_s
    mu = np.sqrt(K_s * d ** 3)

    return build_pgsc(positions, orientations, r_cut=r_cut,
                      J_cut=J_cut, mu=mu)


def w4_2x2_betti(theta_C, theta_alpha, theta_beta,
                  K_s=1.0, tier=1, d=1.0):
    """Return (β₀, β₁) for the 2×2 staggered assembly.

    Parameters
    ----------
    tier : {1, 2}
        1 → nearest-neighbour only (r_cut = d√2·1.001)
        2 → Group-A edges included  (r_cut = d√5·1.001)
    """
    if tier == 1:
        r_cut = d * np.sqrt(2.0) * 1.001
    elif tier == 2:
        r_cut = d * np.sqrt(5.0) * 1.001
    else:
        raise ValueError(f"tier must be 1 or 2, got {tier!r}")
    _, beta0, beta1 = w4_2x2_pgsc(theta_C, theta_alpha, theta_beta,
                                    K_s=K_s, r_cut=r_cut, d=d)
    return beta0, beta1
