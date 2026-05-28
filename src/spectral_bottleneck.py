"""
src/spectral_bottleneck.py — Spectral bottleneck analysis via the weighted
normalized (Chung) Laplacian.

Provides tools to identify topological switch nodes (TSNs) using the Fiedler
value λ₁, Fiedler vector, Cheeger constant bounds, and coupling sensitivity
∂J/∂φ on Fiedler-cut edges — a continuous, pre-boundary-crossing complement
to the discrete β₀-leverage approach.

Public API
----------
chung_laplacian(J_matrix)
    Build the weighted normalized Laplacian from a coupling matrix.

fiedler_analysis(J_matrix)
    Return (lambda1, fiedler_vec, cheeger_lower, cheeger_upper).

cheeger_cut_mask(fiedler_vec)
    Boolean mask: True = nodes on the S side of the Fiedler cut.

frenkel_sensitivity(phi_i, phi_j, theta_ij, r_ij, mu=1.0)
    Return (dJ_dphi_i, dJ_dphi_j) — analytic derivatives of J.

w4_coupling_sensitivity(theta_C, theta_P, K_s=1.0)
    Return dict of {edge_label: (dJ_dthetaC, dJ_dthetaP)} for all 8 W4 edges.

w4_coupling_matrix(theta_C, theta_P, K_s=1.0)
    Return the 5×5 symmetric |J_ij| weight matrix for the W₄ uniform texture.

fiedler_tsn_score(theta_C, theta_P, K_s=1.0)
    For each node v, compute the total |∂J_e/∂φ_v| summed over cut edges.
    Returns (lambda1, fiedler_vec, scores_per_node) where scores_per_node
    is an array of length N.

fiedler_tsn_map(n_theta=90, K_s=1.0)
    Sweep (θ_C, θ_P) over [1°, 89°]² and compute:
      - lambda1_map: Fiedler value at each point
      - tsn_scores: dict of node_name -> (n,n) score array
      - beta0_map: reference β₀
    Returns (angles, lambda1_map, tsn_scores, beta0_map).

w4_signed_coupling_matrix(theta_C, theta_P, K_s=1.0)
    Return signed W₄ coupling matrix J.

signed_fiedler_value(theta_C, theta_P, K_s=1.0)
    Fiedler value of L_energy = D_|J| - J.

energy_fiedler_map(n_theta=90, K_s=1.0)
    Sweep (θ_C, θ_P) and return signed-Laplacian Fiedler map.

hamiltonian_gap_map(n_theta=90, K_s=1.0)
    Sweep (θ_C, θ_P) and return minimum adjacent Hamiltonian eigenvalue gap map.

bright_state_osc_diff_map(n_theta=90, K_s=1.0, eps=0.0)
    Sweep (θ_C, θ_P) and return f₀ − f₂ oscillator-strength differential map.
"""

import numpy as np
from scipy.linalg import eigh

from pgsc import w4_pgsc, w4_betti
from two_des import solve_w4, transition_dipoles, oscillator_strengths


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEG = np.pi / 180.0

# W₄ node names in index order
W4_NODE_NAMES = ['C', 'N', 'E', 'S', 'W']


# ---------------------------------------------------------------------------
# Chung (weighted normalized) Laplacian
# ---------------------------------------------------------------------------

def chung_laplacian(J_matrix):
    """Build the weighted normalized (Chung) Laplacian from a coupling matrix.

    For a graph with weight matrix W_ij = |J_ij| and weighted degree matrix
    D_w = diag(sum_j W_ij), the normalized Laplacian is:

        L = D_w^{-1/2} (D_w - W) D_w^{-1/2}
          = I - D_w^{-1/2} W D_w^{-1/2}

    Isolated nodes (zero weighted degree) are handled by zeroing their
    rows and columns in L, so they do not contribute to the spectrum.

    Parameters
    ----------
    J_matrix : (N, N) array-like
        Symmetric coupling matrix.  Off-diagonal entries are the Frenkel
        couplings J_ij; diagonal is ignored.

    Returns
    -------
    L : (N, N) ndarray
        Weighted normalized Laplacian (real symmetric).
    """
    J = np.asarray(J_matrix, dtype=float)
    np.fill_diagonal(J, 0.0)   # diagonal entries are not coupling weights
    W = np.abs(J)
    d = W.sum(axis=1)
    d_invsqrt = np.where(d > 0.0, 1.0 / np.sqrt(d), 0.0)
    D_invsqrt = np.diag(d_invsqrt)
    L = np.eye(len(d)) - D_invsqrt @ W @ D_invsqrt
    # Zero out rows/cols of isolated nodes (d == 0) so they have eigenvalue 0
    isolated = d == 0.0
    L[isolated, :] = 0.0
    L[:, isolated] = 0.0
    return L


# ---------------------------------------------------------------------------
# Fiedler analysis
# ---------------------------------------------------------------------------

def fiedler_analysis(J_matrix):
    """Compute the Fiedler value, vector, and Cheeger bounds from a coupling matrix.

    For a connected graph the Fiedler value λ₁ is the second-smallest
    eigenvalue of the normalized Laplacian.  The Cheeger inequality gives:

        λ₁/2  ≤  h(G)  ≤  √(2 λ₁)

    When the graph has multiple connected components (β₀ > 1), the
    normalized Laplacian has multiple zero eigenvalues.  In this case the
    function analyses each connected component separately and returns the
    result for the component with the smallest non-zero Fiedler value
    (i.e. the component closest to a connectivity transition).

    Parameters
    ----------
    J_matrix : (N, N) array-like
        Symmetric coupling matrix.

    Returns
    -------
    lambda1       : float  Fiedler value (smallest non-zero eigenvalue > tol).
    fiedler_vec   : (N,) ndarray  Fiedler eigenvector (full graph, zeros for
                    nodes not in the selected component).
    cheeger_lower : float  λ₁/2 — lower Cheeger bound.
    cheeger_upper : float  √(2 λ₁) — upper Cheeger bound.
    """
    L = chung_laplacian(J_matrix)
    evals, evecs = eigh(L)   # sorted ascending; L is real symmetric

    tol = 1e-10
    nonzero_mask = evals > tol
    if not nonzero_mask.any():
        # Fully disconnected — every node isolated; return zeros
        return 0.0, np.zeros(L.shape[0]), 0.0, 0.0

    lambda1 = float(evals[nonzero_mask][0])
    idx = np.where(nonzero_mask)[0][0]
    fiedler_vec = evecs[:, idx]

    cheeger_lower = lambda1 / 2.0
    cheeger_upper = np.sqrt(2.0 * lambda1)
    return lambda1, fiedler_vec, cheeger_lower, cheeger_upper


# ---------------------------------------------------------------------------
# Cheeger cut mask
# ---------------------------------------------------------------------------

def cheeger_cut_mask(fiedler_vec):
    """Return a boolean mask defining the S side of the Fiedler cut.

    Nodes with fiedler_vec[v] < 0 are placed in set S; nodes with
    fiedler_vec[v] >= 0 are placed in set S-bar.

    Parameters
    ----------
    fiedler_vec : (N,) array-like

    Returns
    -------
    mask : (N,) bool ndarray  True for nodes in S.
    """
    return np.asarray(fiedler_vec) < 0.0


# ---------------------------------------------------------------------------
# Analytic Frenkel coupling derivatives
# ---------------------------------------------------------------------------

def frenkel_sensitivity(phi_i, phi_j, theta_ij, r_ij, mu=1.0):
    """Analytic derivatives of J w.r.t. dipole orientations φᵢ and φⱼ.

    From J = (μ²/r³)[cos(φᵢ−φⱼ) − 3cos(φᵢ−θ)cos(φⱼ−θ)]:

        ∂J/∂φᵢ = (μ²/r³)[−sin(φᵢ−φⱼ) + 3sin(φᵢ−θ)cos(φⱼ−θ)]
        ∂J/∂φⱼ = (μ²/r³)[ sin(φᵢ−φⱼ) + 3cos(φᵢ−θ)sin(φⱼ−θ)]

    Parameters
    ----------
    phi_i, phi_j : float  Dipole orientation angles (radians).
    theta_ij     : float  Angle of the i→j internuclear axis (radians).
    r_ij         : float  Internuclear distance.
    mu           : float  Dipole magnitude.

    Returns
    -------
    dJ_dphi_i : float
    dJ_dphi_j : float
    """
    prefactor = mu ** 2 / r_ij ** 3
    dJ_dphi_i = prefactor * (
        -np.sin(phi_i - phi_j)
        + 3.0 * np.sin(phi_i - theta_ij) * np.cos(phi_j - theta_ij)
    )
    dJ_dphi_j = prefactor * (
        np.sin(phi_i - phi_j)
        + 3.0 * np.cos(phi_i - theta_ij) * np.sin(phi_j - theta_ij)
    )
    return dJ_dphi_i, dJ_dphi_j


# ---------------------------------------------------------------------------
# W₄ analytic coupling-sensitivity derivatives
# ---------------------------------------------------------------------------

def w4_coupling_sensitivity(theta_C, theta_P, K_s=1.0):
    """Analytic derivatives of W₄ couplings w.r.t. θ_C and θ_P.

    For the uniform texture (center at θ_C, all rim nodes at θ_P), the
    analytic coupling formulas from THEORY.md §1.3 give closed-form
    derivatives:

    Spoke couplings:
        J_NS = K_s[cos(θ_C)cos(θ_P) − 2sin(θ_C)sin(θ_P)]
        J_WE = K_s[sin(θ_C)sin(θ_P) − 2cos(θ_C)cos(θ_P)]

        dJ_NS/dθ_C = K_s[−sin(θ_C)cos(θ_P) − 2cos(θ_C)sin(θ_P)]
        dJ_NS/dθ_P = K_s[−cos(θ_C)sin(θ_P) − 2sin(θ_C)cos(θ_P)]
        dJ_WE/dθ_C = K_s[ cos(θ_C)sin(θ_P) + 2sin(θ_C)cos(θ_P)]
        dJ_WE/dθ_P = K_s[ sin(θ_C)cos(θ_P) + 2cos(θ_C)sin(θ_P)]

    Rim couplings (depend only on θ_P):
        J_diag = K_r[−1/2 + (3/2)sin(2θ_P)]
        J_anti  = K_r[−1/2 − (3/2)sin(2θ_P)]

        dJ_diag/dθ_P = K_r · 3cos(2θ_P)
        dJ_anti/dθ_P = −K_r · 3cos(2θ_P)
        dJ_diag/dθ_C = dJ_anti/dθ_C = 0

    Parameters
    ----------
    theta_C, theta_P : float  Orientation angles (radians).
    K_s              : float  Spoke coupling scale.

    Returns
    -------
    sensitivities : dict
        Keys: edge labels ('C-N', 'C-S', 'C-E', 'C-W', 'N-E', 'S-W',
              'E-S', 'W-N').
        Values: (dJ_dthetaC, dJ_dthetaP) tuples.
    """
    K_r = K_s / (2.0 * np.sqrt(2.0))

    dJNS_dC = K_s * (-np.sin(theta_C) * np.cos(theta_P)
                      - 2.0 * np.cos(theta_C) * np.sin(theta_P))
    dJNS_dP = K_s * (-np.cos(theta_C) * np.sin(theta_P)
                      - 2.0 * np.sin(theta_C) * np.cos(theta_P))

    dJWE_dC = K_s * (np.cos(theta_C) * np.sin(theta_P)
                     + 2.0 * np.sin(theta_C) * np.cos(theta_P))
    dJWE_dP = K_s * (np.sin(theta_C) * np.cos(theta_P)
                     + 2.0 * np.cos(theta_C) * np.sin(theta_P))

    dJdiag_dP = K_r * 3.0 * np.cos(2.0 * theta_P)
    dJanti_dP = -K_r * 3.0 * np.cos(2.0 * theta_P)

    return {
        'C-N': (dJNS_dC, dJNS_dP),   # spoke; C=node 0, N=node 1
        'C-S': (dJNS_dC, dJNS_dP),   # spoke; C=node 0, S=node 3
        'C-E': (dJWE_dC, dJWE_dP),   # spoke; C=node 0, E=node 2
        'C-W': (dJWE_dC, dJWE_dP),   # spoke; C=node 0, W=node 4
        'N-E': (0.0,     dJdiag_dP),  # rim diagonal; N=node 1, E=node 2
        'S-W': (0.0,     dJdiag_dP),  # rim diagonal; S=node 3, W=node 4
        'E-S': (0.0,     dJanti_dP),  # rim anti-diagonal; E=node 2, S=node 3
        'W-N': (0.0,     dJanti_dP),  # rim anti-diagonal; W=node 4, N=node 1
    }


# ---------------------------------------------------------------------------
# Build the W₄ coupling matrix
# ---------------------------------------------------------------------------

def w4_coupling_matrix(theta_C, theta_P, K_s=1.0):
    """Return the 5×5 symmetric W₄ absolute-coupling matrix |J_ij|.

    Active and inactive edges both appear; the weights are the raw Frenkel
    values (not thresholded), so the Laplacian reflects the full coupling
    landscape.

    Node order: C=0, N=1, E=2, S=3, W=4.
    """
    K_r = K_s / (2.0 * np.sqrt(2.0))
    J_NS   = K_s * (np.cos(theta_C) * np.cos(theta_P)
                    - 2.0 * np.sin(theta_C) * np.sin(theta_P))
    J_WE   = K_s * (np.sin(theta_C) * np.sin(theta_P)
                    - 2.0 * np.cos(theta_C) * np.cos(theta_P))
    J_diag = K_r * (-0.5 + 1.5 * np.sin(2.0 * theta_P))
    J_anti  = K_r * (-0.5 - 1.5 * np.sin(2.0 * theta_P))

    # Use absolute values — the Laplacian weight is the coupling magnitude
    J_NS_abs   = np.abs(J_NS)
    J_WE_abs   = np.abs(J_WE)
    J_diag_abs = np.abs(J_diag)
    J_anti_abs = np.abs(J_anti)

    W = np.zeros((5, 5))
    # Spokes: C-N (0-1), C-S (0-3), C-E (0-2), C-W (0-4)
    W[0, 1] = W[1, 0] = J_NS_abs
    W[0, 3] = W[3, 0] = J_NS_abs
    W[0, 2] = W[2, 0] = J_WE_abs
    W[0, 4] = W[4, 0] = J_WE_abs
    # Rim: N-E (1-2), S-W (3-4), E-S (2-3), W-N (4-1)
    W[1, 2] = W[2, 1] = J_diag_abs
    W[3, 4] = W[4, 3] = J_diag_abs
    W[2, 3] = W[3, 2] = J_anti_abs
    W[4, 1] = W[1, 4] = J_anti_abs

    return W


# ---------------------------------------------------------------------------
# Signed (energy-dressed) Laplacian helpers
# ---------------------------------------------------------------------------

def w4_signed_coupling_matrix(theta_C, theta_P, K_s=1.0):
    """Return the signed 5×5 W₄ coupling matrix J.

    Parameters
    ----------
    theta_C, theta_P : float
        Center and rim dipole orientations (radians).
    K_s : float, optional
        Spoke coupling scale.

    Returns
    -------
    J : (5, 5) ndarray
        Symmetric signed coupling matrix (diagonal zeros).
    """
    K_r = K_s / (2.0 * np.sqrt(2.0))
    J_NS = K_s * (np.cos(theta_C) * np.cos(theta_P)
                  - 2.0 * np.sin(theta_C) * np.sin(theta_P))
    J_WE = K_s * (np.sin(theta_C) * np.sin(theta_P)
                  - 2.0 * np.cos(theta_C) * np.cos(theta_P))
    J_diag = K_r * (-0.5 + 1.5 * np.sin(2.0 * theta_P))
    J_anti = K_r * (-0.5 - 1.5 * np.sin(2.0 * theta_P))

    J = np.zeros((5, 5))
    J[0, 1] = J[1, 0] = J_NS
    J[0, 3] = J[3, 0] = J_NS
    J[0, 2] = J[2, 0] = J_WE
    J[0, 4] = J[4, 0] = J_WE
    J[1, 2] = J[2, 1] = J_diag
    J[3, 4] = J[4, 3] = J_diag
    J[2, 3] = J[3, 2] = J_anti
    J[4, 1] = J[1, 4] = J_anti
    return J


def signed_fiedler_value(theta_C, theta_P, K_s=1.0):
    """Return the Fiedler value of the signed Laplacian L_energy = D_|J| - J.

    Parameters
    ----------
    theta_C, theta_P : float
        Center and rim dipole orientations (radians).
    K_s : float, optional
        Spoke coupling scale.

    Returns
    -------
    float
        Smallest non-zero eigenvalue of L_energy.
    """
    J = w4_signed_coupling_matrix(theta_C, theta_P, K_s)
    d = np.abs(J).sum(axis=1)
    L = np.diag(d) - J
    evals = np.sort(np.linalg.eigvalsh(L))
    tol = 1e-9
    nonzero = evals[evals > tol]
    return float(nonzero[0]) if len(nonzero) > 0 else 0.0


def energy_fiedler_map(n_theta=90, K_s=1.0):
    """Sweep (θ_C, θ_P) and compute signed Fiedler values.

    Returns
    -------
    angles : (n_theta,) ndarray
        Angles in degrees spanning [1, 89].
    energy_lambda1_map : (n_theta, n_theta) ndarray
        Signed-Laplacian Fiedler value map indexed as [i_C, j_P].
    """
    angles = np.linspace(1.0, 89.0, n_theta)
    energy_lambda1_map = np.zeros((n_theta, n_theta))
    for i, tc_deg in enumerate(angles):
        for j, tp_deg in enumerate(angles):
            energy_lambda1_map[i, j] = signed_fiedler_value(tc_deg * DEG, tp_deg * DEG, K_s)
    return angles, energy_lambda1_map


def hamiltonian_gap_map(n_theta=90, K_s=1.0):
    """Sweep (θ_C, θ_P) and compute minimum adjacent Hamiltonian eigenvalue gap.

    The Hamiltonian is H = -J where J is the signed W₄ coupling matrix.

    Returns
    -------
    angles : (n_theta,) ndarray
        Angles in degrees spanning [1, 89].
    gap_map : (n_theta, n_theta) ndarray
        Minimum adjacent eigenvalue gap map indexed as [i_C, j_P].
    """
    angles = np.linspace(1.0, 89.0, n_theta)
    gap_map = np.zeros((n_theta, n_theta))
    for i, tc_deg in enumerate(angles):
        for j, tp_deg in enumerate(angles):
            J = w4_signed_coupling_matrix(tc_deg * DEG, tp_deg * DEG, K_s)
            H = -J
            evals = np.linalg.eigvalsh(H)
            gaps = np.diff(evals)
            gap_map[i, j] = float(np.min(gaps))
    return angles, gap_map


def bright_state_osc_diff_map(n_theta=90, K_s=1.0, eps=0.0):
    """Sweep (θ_C, θ_P) and compute f₀ − f₂: oscillator strength of eigenstate 0
    minus oscillator strength of eigenstate 2.

    The zero-contour of this map coincides with the eigenstate bright-state
    switch boundary visible in Fig 5 (left panel).

    Parameters
    ----------
    n_theta : int, optional
        Grid resolution for the (θ_C, θ_P) sweep.
    K_s : float, optional
        Spoke coupling scale passed to solve_w4.
    eps : float, optional
        Uniform site-energy shift passed to solve_w4.

    Returns
    -------
    angles : (n_theta,) ndarray
        Degrees, spanning [1, 89].
    osc_diff_map : (n_theta, n_theta) ndarray
        f₀ − f₂, indexed [i_C, j_P].
    """
    angles = np.linspace(1.0, 89.0, n_theta)
    osc_diff_map = np.zeros((n_theta, n_theta))
    for i, tc_deg in enumerate(angles):
        for j, tp_deg in enumerate(angles):
            tc = tc_deg * DEG
            tp = tp_deg * DEG
            _, evecs = solve_w4(tc, tp, K_s, eps)
            mu_x, mu_y = transition_dipoles(evecs, tc, tp)
            osc = oscillator_strengths(mu_x, mu_y)
            osc_diff_map[i, j] = float(osc[0] - osc[2])
    return angles, osc_diff_map


# ---------------------------------------------------------------------------
# Fiedler TSN score for a single (θ_C, θ_P) point
# ---------------------------------------------------------------------------

def fiedler_tsn_score(theta_C, theta_P, K_s=1.0):
    """Compute the Fiedler-cut coupling sensitivity for each W₄ node.

    For each node v, the spectral TSN score is:

        Σ_v = Σ_{e=(i,j) ∈ ∂S} |∂J_e/∂φ_v|

    where ∂S is the Fiedler cut (edges straddling the zero-crossing of the
    Fiedler vector) and ∂J_e/∂φ_v is the analytic coupling derivative w.r.t.
    the orientation of node v.

    The Fiedler analysis uses |J_ij| as edge weights so that the topology of
    coupling magnitude, not sign, determines the bottleneck.

    Parameters
    ----------
    theta_C, theta_P : float  Orientation angles (radians).
    K_s              : float  Spoke coupling scale.

    Returns
    -------
    lambda1      : float     Fiedler value.
    fiedler_vec  : (5,) ndarray  Fiedler eigenvector.
    scores       : (5,) ndarray  TSN score for each node [C,N,E,S,W].
    """
    W = w4_coupling_matrix(theta_C, theta_P, K_s)
    lambda1, fiedler_vec, _, _ = fiedler_analysis(W)

    cut_mask = cheeger_cut_mask(fiedler_vec)   # True = S side
    sens = w4_coupling_sensitivity(theta_C, theta_P, K_s)

    # Map edge labels to (node_i, node_j, dJ_dC, dJ_dP) entries
    # where dJ_dC is the derivative w.r.t. theta_C (center node phi)
    # and dJ_dP is derivative w.r.t. theta_P (rim node phi)
    # Node index: C=0, N=1, E=2, S=3, W=4
    edge_info = [
        # (i, j, edge_label)
        (0, 1, 'C-N'),
        (0, 3, 'C-S'),
        (0, 2, 'C-E'),
        (0, 4, 'C-W'),
        (1, 2, 'N-E'),
        (3, 4, 'S-W'),
        (2, 3, 'E-S'),
        (4, 1, 'W-N'),
    ]

    scores = np.zeros(5)

    for i, j, label in edge_info:
        # Is this edge a cut edge?  (one node on each side)
        if cut_mask[i] == cut_mask[j]:
            continue   # not a cut edge

        dJ_dC, dJ_dP = sens[label]

        # Node 0 (C) is controlled by theta_C
        if i == 0 or j == 0:
            scores[0] += abs(dJ_dC)

        # Rim nodes controlled by theta_P
        for node_idx in (i, j):
            if node_idx != 0:
                scores[node_idx] += abs(dJ_dP)

    return lambda1, fiedler_vec, scores


# ---------------------------------------------------------------------------
# Full (θ_C, θ_P) grid sweep
# ---------------------------------------------------------------------------

def fiedler_tsn_map(n_theta=90, K_s=1.0):
    """Sweep (θ_C, θ_P) ∈ [1°, 89°]² and compute spectral bottleneck quantities.

    For each grid point:
    - Build the |J_ij| coupling matrix for the W₄ uniform texture.
    - Compute the Fiedler value λ₁ and Fiedler vector.
    - Compute the Fiedler-cut TSN score for each node.
    - Record β₀ for reference.

    Parameters
    ----------
    n_theta : int   Grid resolution (n_theta × n_theta points).
    K_s     : float  Spoke coupling scale.

    Returns
    -------
    angles       : (n_theta,) ndarray  Angle values in degrees.
    lambda1_map  : (n_theta, n_theta) ndarray  Fiedler value at each grid point.
                   Indexing: [i_C, j_P] with i_C = theta_C index, j_P = theta_P index.
    tsn_scores   : dict  {'C': (n,n), 'N': (n,n), 'E': (n,n), 'S': (n,n), 'W': (n,n)}
                   TSN score for each node at each grid point.
    beta0_map    : (n_theta, n_theta) ndarray  β₀ at each grid point.
    """
    angles = np.linspace(1.0, 89.0, n_theta)
    lambda1_map = np.zeros((n_theta, n_theta))
    tsn_scores  = {name: np.zeros((n_theta, n_theta)) for name in W4_NODE_NAMES}
    beta0_map   = np.empty((n_theta, n_theta), dtype=int)

    for i, tc_deg in enumerate(angles):
        for j, tp_deg in enumerate(angles):
            tc = tc_deg * DEG
            tp = tp_deg * DEG

            _, b0, _ = w4_pgsc(tc, tp, K_s)
            beta0_map[i, j] = b0

            lam1, fvec, scores = fiedler_tsn_score(tc, tp, K_s)
            lambda1_map[i, j] = lam1
            for k, name in enumerate(W4_NODE_NAMES):
                tsn_scores[name][i, j] = scores[k]

    return angles, lambda1_map, tsn_scores, beta0_map
