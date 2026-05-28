"""
src/phase_diagram.py — Phase diagram sweep and topological leverage utilities.

Provides:
  sweep_single_tile(n_theta)   →  (θ_C, θ_P, β₀, β₁) arrays over [0°, 90°]²
  topological_leverage(n_theta) →  leverage map for each W₄ node  [deprecated]
  sweep_2x2_tiling(n_alpha, n_beta, theta_C, tier) →  (θ_α, θ_β, β₀, β₁) arrays
  classify_region_numeric(tc, tp) → region label ('A'–'F')

These are the numerical counterparts of the analytical phase diagram in
THEORY.md §4, used to produce Figures 2, 3, 5, and 6 of Paper 1.

Note on TSN identification
--------------------------
The `topological_leverage` function implements a finite-difference β₀-sensitivity
approach that is nonzero only right at phase boundaries.  For a continuous,
pre-boundary-crossing alternative that also identifies the spatial location of
the bottleneck cut, see `spectral_bottleneck.fiedler_tsn_map()`.  That module
uses the weighted normalized (Chung) Laplacian and Fiedler vector to provide a
spectral TSN score across the entire phase diagram, not just at phase boundaries.
"""

import numpy as np
from pgsc import w4_betti, w4_2x2_betti


# ---------------------------------------------------------------------------
# Phase boundary helpers (THEORY.md §4.1)
# ---------------------------------------------------------------------------

L1_DEG = 0.5 * np.degrees(np.arcsin(1.0 / 3.0))   # ≈ 9.74°
L2_DEG = 90.0 - L1_DEG                              # ≈ 80.26°

DEG = np.pi / 180.0


def classify_region_numeric(theta_C_deg, theta_P_deg):
    """Return the phase-diagram region label ('A'–'F') for a given orientation.

    Implements the six-region classification from THEORY.md §4.2 using the
    exact boundary conditions (not Betti numbers), making it independent of
    J_cut and useful as a ground-truth reference.

    Parameters
    ----------
    theta_C_deg, theta_P_deg : float  Angles in degrees.

    Returns
    -------
    region : str  One of 'A', 'B', 'C', 'D', 'E', 'F'.
    """
    tp = theta_P_deg
    tc = theta_C_deg
    tan_prod = np.tan(tc * DEG) * np.tan(tp * DEG)
    rim_active = (L1_DEG < tp < L2_DEG)

    if rim_active:
        if tan_prod < 0.5:
            return 'A'
        elif tan_prod < 2.0:
            return 'B'
        else:
            return 'C'
    else:
        if tan_prod < 0.5:
            return 'D'
        elif tan_prod < 2.0:
            return 'E'
        else:
            return 'F'


# ---------------------------------------------------------------------------
# Single-tile sweep (Figures 2 and 3 of Paper 1)
# ---------------------------------------------------------------------------

def sweep_single_tile(n_theta=180):
    """Sweep (θ_C, θ_P) ∈ [0°, 90°]² and compute β₀ and β₁.

    Parameters
    ----------
    n_theta : int  Grid resolution (n_theta × n_theta points).

    Returns
    -------
    tc_grid   : (n_theta, n_theta) ndarray  θ_C values in degrees.
    tp_grid   : (n_theta, n_theta) ndarray  θ_P values in degrees.
    beta0_map : (n_theta, n_theta) ndarray  β₀ values (int).
    beta1_map : (n_theta, n_theta) ndarray  β₁ values (int).
    """
    angles = np.linspace(0.0, 90.0, n_theta)
    tc_grid, tp_grid = np.meshgrid(angles, angles, indexing='ij')
    beta0_map = np.empty((n_theta, n_theta), dtype=int)
    beta1_map = np.empty((n_theta, n_theta), dtype=int)

    for i, tc_deg in enumerate(angles):
        for j, tp_deg in enumerate(angles):
            b0, b1 = w4_betti(tc_deg * DEG, tp_deg * DEG)
            beta0_map[i, j] = b0
            beta1_map[i, j] = b1

    return tc_grid, tp_grid, beta0_map, beta1_map


# ---------------------------------------------------------------------------
# Topological leverage (Figure 6 of Paper 1)
# ---------------------------------------------------------------------------

def topological_leverage(n_theta=180, K_s=1.0):
    """Compute the topological leverage of each W₄ node across the phase diagram.

    .. deprecated::
        This function computes |Δβ₀|/Δφ, which is nonzero only right at a
        phase boundary and zero everywhere else.  For a more informative
        continuous TSN landscape use `spectral_bottleneck.fiedler_tsn_map()`,
        which provides the Fiedler value λ₁, Fiedler vector, and Cheeger-cut
        coupling sensitivity across the entire phase diagram.

    The leverage of node v at orientation (θ_C, θ_P) is defined as the change
    in β₀ when node v's orientation is perturbed by Δθ = 1° while all other
    nodes are held fixed:

        Λ_v(θ_C, θ_P) = |β₀(perturbed) − β₀(unperturbed)| / Δθ

    Units: changes in β₀ per degree (deg⁻¹).  A leverage of 1 deg⁻¹ means a
    1° rotation causes a unit change in β₀.  Λ_v = 0 for nodes whose rotation
    never crosses a phase boundary; bottleneck nodes near phase boundaries have
    the highest leverage.

    Parameters
    ----------
    n_theta : int  Grid resolution.
    K_s     : float  Coupling scale.

    Returns
    -------
    angles      : (n_theta,) ndarray  Angle values in degrees.
    leverage    : dict  {'C': (n,n) array, 'N': ..., 'E': ..., 'S': ..., 'W': ...}
    beta0_map   : (n_theta, n_theta) ndarray  Reference β₀ at uniform texture.
    """
    from pgsc import w4_pgsc

    angles = np.linspace(1.0, 89.0, n_theta)   # avoid exact 0° and 90° boundaries
    delta_deg = 1.0
    delta = delta_deg * DEG

    leverage = {name: np.zeros((n_theta, n_theta)) for name in 'CNESW'}
    beta0_map = np.empty((n_theta, n_theta), dtype=int)

    for i, tc_deg in enumerate(angles):
        for j, tp_deg in enumerate(angles):
            tc, tp = tc_deg * DEG, tp_deg * DEG

            # Reference Betti numbers (uniform texture)
            _, b0_ref, _ = w4_pgsc(tc, tp, K_s)
            beta0_map[i, j] = b0_ref

            # Perturb center node (C)
            _, b0_p, _ = w4_pgsc(tc + delta, tp, K_s)
            leverage['C'][i, j] = abs(b0_p - b0_ref) / delta_deg

            # Perturb rim nodes (N = S by symmetry, E = W by symmetry)
            _, b0_p, _ = w4_pgsc(tc, tp + delta, K_s)
            lev_rim = abs(b0_p - b0_ref) / delta_deg
            leverage['N'][i, j] = lev_rim
            leverage['S'][i, j] = lev_rim
            leverage['E'][i, j] = lev_rim
            leverage['W'][i, j] = lev_rim

    return angles, leverage, beta0_map


# ---------------------------------------------------------------------------
# 2×2 tiling sweep (Figure 5 of Paper 1)
# ---------------------------------------------------------------------------

def sweep_2x2_tiling(n_alpha=60, n_beta=60, theta_C_deg=45.0, tier=1):
    """Sweep (θ_α, θ_β) ∈ [0°, 90°]² for the 2×2 staggered assembly.

    Parameters
    ----------
    n_alpha, n_beta : int  Grid resolutions.
    theta_C_deg     : float  Fixed center orientation in degrees.
    tier            : {1, 2}  Coupling tier (1 = nearest-neighbour, 2 = Group A).

    Returns
    -------
    ta_grid   : (n_alpha, n_beta) ndarray  θ_α values in degrees.
    tb_grid   : (n_alpha, n_beta) ndarray  θ_β values in degrees.
    beta0_map : (n_alpha, n_beta) ndarray  β₀ values.
    beta1_map : (n_alpha, n_beta) ndarray  β₁ values.
    """
    alphas = np.linspace(0.0, 90.0, n_alpha)
    betas  = np.linspace(0.0, 90.0, n_beta)
    ta_grid, tb_grid = np.meshgrid(alphas, betas, indexing='ij')
    beta0_map = np.empty((n_alpha, n_beta), dtype=int)
    beta1_map = np.empty((n_alpha, n_beta), dtype=int)

    tc = theta_C_deg * DEG
    for i, ta_deg in enumerate(alphas):
        for j, tb_deg in enumerate(betas):
            b0, b1 = w4_2x2_betti(tc, ta_deg * DEG, tb_deg * DEG,
                                    tier=tier)
            beta0_map[i, j] = b0
            beta1_map[i, j] = b1

    return ta_grid, tb_grid, beta0_map, beta1_map
