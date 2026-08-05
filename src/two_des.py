"""
Two-Dimensional Electronic Spectroscopy (2DES) theory for the W4 polarization-topology system.

This module computes:
  - Frenkel coupling constants for the W4 wheel graph
  - The 5x5 Hamiltonian and its block-decomposed forms
  - Eigenstates, transition dipoles, and oscillator strengths
  - Linear absorption spectra
  - Absorptive 2DES spectra (impulsive limit, T=0)
  - Berry phase sign for loops in (theta_C, theta_P) space

Conventions
-----------
Node order throughout: [C=0, N=1, E=2, S=3, W=4]
  C = center chromophore at (0, 0)
  N = north  at (0, +d),  E = east  at (+d, 0)
  S = south  at (0, -d),  W = west  at (-d, 0)

All angles in radians unless noted.

Energies in units of K_s (the spoke coupling scale) unless a value is passed.

Rim edges by label:
  N-E and S-W: J_diag = K_r[-1/2 + (3/2)sin(2*theta_P)]
  E-S and W-N: J_anti = K_r[-1/2 - (3/2)sin(2*theta_P)]
where K_r = K_s / (2*sqrt(2)).

The rim frustration identity J_diag + J_anti = -K_r holds exactly for all theta_P.

The C2 symmetry (180-degree rotation swapping N<->S and E<->W) block-diagonalises H into:
  - 3x3 symmetric block (basis: {(N+S)/sqrt(2), (E+W)/sqrt(2), C})
  - 2x2 antisymmetric block (basis: {(N-S)/sqrt(2), (E-W)/sqrt(2)})

The two antisymmetric eigenstates are permanently dark (zero transition dipole)
for any orientation of a uniform texture tile.
"""

import numpy as np
from scipy.linalg import eigh as sp_eigh

DEGENERACY_GAP_TOL = 1e-7

# ---------------------------------------------------------------------------
# Coupling constants
# ---------------------------------------------------------------------------

def w4_couplings(theta_C, theta_P, K_s=1.0):
    """
    Compute the four independent Frenkel coupling constants for a uniform W4 tile.

    Parameters
    ----------
    theta_C : float
        Center chromophore orientation angle (radians).
    theta_P : float
        Rim chromophore orientation angle (radians); same for all four rim nodes.
    K_s : float
        Spoke coupling scale = mu^2 / d^3.

    Returns
    -------
    J_NS : float
        Spoke coupling (C-N = C-S).
    J_WE : float
        Spoke coupling (C-E = C-W).
    J_diag : float
        Diagonal rim coupling (N-E = S-W).
    J_anti : float
        Anti-diagonal rim coupling (E-S = W-N).
    """
    K_r = K_s / (2.0 * np.sqrt(2.0))
    J_NS = K_s * (np.cos(theta_C) * np.cos(theta_P) - 2.0 * np.sin(theta_C) * np.sin(theta_P))
    J_WE = K_s * (np.sin(theta_C) * np.sin(theta_P) - 2.0 * np.cos(theta_C) * np.cos(theta_P))
    J_diag = K_r * (-0.5 + 1.5 * np.sin(2.0 * theta_P))
    J_anti = K_r * (-0.5 - 1.5 * np.sin(2.0 * theta_P))
    return J_NS, J_WE, J_diag, J_anti


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

def w4_hamiltonian(theta_C, theta_P, K_s=1.0, epsilon=0.0, delta_epsilon=0.0):
    """
    Construct the full 5x5 W4 Frenkel Hamiltonian.

    Node order: [C=0, N=1, E=2, S=3, W=4].
    Rim edges: N-E (J_diag), S-W (J_diag), E-S (J_anti), W-N (J_anti).

    Parameters
    ----------
    theta_C : float
        Center orientation angle (radians).
    theta_P : float
        Rim orientation angle (radians).
    K_s : float
        Spoke coupling scale.
    epsilon : float
        Rim site energy.
    delta_epsilon : float
        Center site energy offset: epsilon_C = epsilon + delta_epsilon.

    Returns
    -------
    H : (5, 5) ndarray, real symmetric.
    """
    J_NS, J_WE, J_diag, J_anti = w4_couplings(theta_C, theta_P, K_s)
    epsilon_C = epsilon + delta_epsilon

    H = np.diag([epsilon_C, epsilon, epsilon, epsilon, epsilon])

    # Spoke edges
    H[0, 1] = H[1, 0] = J_NS   # C-N
    H[0, 3] = H[3, 0] = J_NS   # C-S
    H[0, 2] = H[2, 0] = J_WE   # C-E
    H[0, 4] = H[4, 0] = J_WE   # C-W

    # Rim edges
    H[1, 2] = H[2, 1] = J_diag  # N-E
    H[3, 4] = H[4, 3] = J_diag  # S-W
    H[2, 3] = H[3, 2] = J_anti  # E-S
    H[1, 4] = H[4, 1] = J_anti  # W-N

    return H


def symmetric_block(theta_C, theta_P, K_s=1.0, delta_epsilon=0.0):
    """
    Construct the 3x3 symmetric block of the W4 Hamiltonian.

    Basis: {(|N>+|S>)/sqrt(2),  (|E>+|W>)/sqrt(2),  |C>}.
    Energies measured from epsilon_rim.

    The off-diagonal rim coupling is always -K_r (rim frustration identity:
    J_diag + J_anti = -K_r for all theta_P).

    Returns
    -------
    M : (3, 3) ndarray, real symmetric.
    """
    J_NS, J_WE, _, _ = w4_couplings(theta_C, theta_P, K_s)
    K_r = K_s / (2.0 * np.sqrt(2.0))

    M = np.array([
        [0.0,              -K_r,              np.sqrt(2.0) * J_NS],
        [-K_r,              0.0,              np.sqrt(2.0) * J_WE],
        [np.sqrt(2.0) * J_NS, np.sqrt(2.0) * J_WE, delta_epsilon  ],
    ])
    return M


def antisymmetric_block(theta_P, K_s=1.0):
    """
    Construct the 2x2 antisymmetric block of the W4 Hamiltonian.

    Basis: {(|N>-|S>)/sqrt(2),  (|E>-|W>)/sqrt(2)}.
    Energies measured from epsilon_rim.

    The exact eigenvalues are +/-3*K_r*sin(2*theta_P), independent of theta_C.

    Returns
    -------
    M : (2, 2) ndarray.
    """
    K_r = K_s / (2.0 * np.sqrt(2.0))
    J_diff = 3.0 * K_r * np.sin(2.0 * theta_P)   # = J_diag - J_anti

    return np.array([
        [0.0,    J_diff],
        [J_diff, 0.0   ],
    ])


# ---------------------------------------------------------------------------
# Eigenvalue / eigenvector solver
# ---------------------------------------------------------------------------

def solve_w4(theta_C, theta_P, K_s=1.0, epsilon=0.0, delta_epsilon=0.0):
    """
    Diagonalise the full 5x5 W4 Hamiltonian.

    Returns
    -------
    eigenvalues : (5,) ndarray, sorted ascending.
    eigenvectors : (5, 5) ndarray, columns are normalised eigenvectors.
    """
    H = w4_hamiltonian(theta_C, theta_P, K_s, epsilon, delta_epsilon)
    return sp_eigh(H)


# ---------------------------------------------------------------------------
# Transition dipoles and oscillator strengths
# ---------------------------------------------------------------------------

def transition_dipoles(eigenvectors, theta_C, theta_P, mu=1.0):
    """
    Compute the x and y transition dipole components for all five eigenstates.

    The transition dipole of eigenstate alpha is
      mu^alpha = sum_i  v_i^alpha * mu_i
    where mu_i = mu*(cos phi_i, sin phi_i) for node i with orientation phi_i.

    For uniform texture:  orientations = [theta_C, theta_P, theta_P, theta_P, theta_P].

    Parameters
    ----------
    eigenvectors : (5, 5) ndarray, columns are eigenvectors.
    theta_C, theta_P : float, orientation angles (radians).
    mu : float, common dipole magnitude.

    Returns
    -------
    mu_x : (5,) ndarray.
    mu_y : (5,) ndarray.
    """
    orientations = np.array([theta_C, theta_P, theta_P, theta_P, theta_P])
    site_mu_x = mu * np.cos(orientations)
    site_mu_y = mu * np.sin(orientations)

    # eigenvectors.T @ site_mu maps site dipoles onto eigenstate dipoles
    mu_x = eigenvectors.T @ site_mu_x
    mu_y = eigenvectors.T @ site_mu_y

    return mu_x, mu_y


def oscillator_strengths(mu_x, mu_y):
    """
    Compute oscillator strengths |mu^alpha|^2 for all eigenstates.

    Returns
    -------
    osc : (N,) ndarray.
    """
    return mu_x ** 2 + mu_y ** 2


def read_crosspeak_signs(energies, mu_x, mu_y, theta_C=None, theta_P=None, bright_tol_rel=1e-8):
    """
    Read coupling-sign tells from analytically defined XYXY cross-peak amplitudes.

    The two physically selected cross-peaks are:
      (E_low, E_mid)   -> sign(J_NS)
      (E_mid, E_high)  -> sign(J_WE)

    Parameters
    ----------
    energies : (N,) ndarray
        Eigenvalues from `solve_w4`.
    mu_x, mu_y : (N,) ndarray
        Transition dipole components from `transition_dipoles`.
    theta_C, theta_P : float or None
        Optional orientation angles. If provided, they are used to robustly identify
        the three symmetric states and return coupling signs from `w4_couplings`.
        If omitted, signs are read directly from selected XYXY amplitudes.
    bright_tol_rel : float
        Relative oscillator-strength threshold used to identify bright states.

    Returns
    -------
    dict with keys:
      'sign_J_NS', 'sign_J_WE' : int (+1 or -1)
      'amp_low_mid', 'amp_mid_high' : float
      'E_low', 'E_mid', 'E_high' : float

    Raises
    ------
    ValueError
        If fewer than three bright states are found (e.g., near CI degeneracy),
        or if either selected cross-peak amplitude is too close to zero to assign
        a robust sign.
    """
    energies = np.asarray(energies, dtype=float)
    mu_x = np.asarray(mu_x, dtype=float)
    mu_y = np.asarray(mu_y, dtype=float)

    if not (energies.shape == mu_x.shape == mu_y.shape):
        raise ValueError(
            "energies, mu_x, and mu_y must have identical 1D shapes."
        )

    if energies.ndim != 1:
        raise ValueError("energies, mu_x, and mu_y must be 1D arrays.")

    order = np.argsort(energies)
    E = energies[order]
    mx = mu_x[order]
    my = mu_y[order]
    osc = oscillator_strengths(mx, my)
    bright_scale = float(np.max(osc)) if osc.size else 0.0
    bright_tol = max(1e-14, bright_tol_rel * bright_scale)

    symmetric_idx = None
    sign_J_NS = None
    sign_J_WE = None

    if (theta_C is None) != (theta_P is None):
        raise ValueError(
            "theta_C and theta_P must both be provided or both omitted, because "
            "they are used together for robust symmetric-state identification."
        )

    if theta_C is not None and theta_P is not None:
        tc = float(theta_C)
        tp = float(theta_P)
        evals_ctx, evecs_ctx = solve_w4(tc, tp)

        if (
            evals_ctx.shape == E.shape
            and not np.allclose(np.sort(evals_ctx), E, rtol=1e-8, atol=1e-10)
        ):
            raise ValueError(
                "Provided theta_C/theta_P are inconsistent with input energies."
            )

        evecs_ctx = evecs_ctx[:, np.argsort(evals_ctx)]
        parity_sym = (
            np.abs(evecs_ctx[1, :] - evecs_ctx[3, :])
            + np.abs(evecs_ctx[2, :] - evecs_ctx[4, :])
        )
        symmetric_idx = np.argsort(parity_sym)[:3]
        symmetric_idx = symmetric_idx[np.argsort(E[symmetric_idx])]

        J_NS, J_WE, _, _ = w4_couplings(tc, tp)
        sign_J_NS = 1 if J_NS >= 0.0 else -1
        sign_J_WE = 1 if J_WE >= 0.0 else -1

    if symmetric_idx is None:
        bright_idx = np.where(osc > bright_tol)[0]
        if bright_idx.size < 3:
            raise ValueError(
                "Cannot read cross-peak signs: found fewer than 3 bright states "
                f"(found {bright_idx.size}). This typically indicates a CI/degeneracy "
                "where one bright-state partner becomes dark."
            )
        symmetric_idx = bright_idx[:3]

    low_i, mid_i, high_i = symmetric_idx
    upper_gap = float(abs(E[high_i] - E[mid_i]))

    n_bright_symmetric = int((osc[symmetric_idx] > bright_tol).sum())
    if n_bright_symmetric < 3 and upper_gap < DEGENERACY_GAP_TOL:
        raise ValueError(
            "Cannot read cross-peak signs: found fewer than 3 bright states "
            "with upper-state degeneracy "
            "(CI-like bright/dark partner)."
        )

    amp_low_mid = float((mx[low_i] * my[low_i]) * (mx[mid_i] * my[mid_i]))
    amp_mid_high = float((mx[mid_i] * my[mid_i]) * (mx[high_i] * my[high_i]))

    if sign_J_NS is None:
        amp_scale = max(1.0, float(np.max(np.abs(mx * my))) ** 2)
        amp_tol = 1e-12 * amp_scale
        if abs(amp_low_mid) <= amp_tol or abs(amp_mid_high) <= amp_tol:
            raise ValueError(
                "Cannot assign a robust cross-peak sign because one selected XYXY "
                "cross-peak amplitude is ~0 (likely near a phase boundary)."
            )
        sign_J_NS = 1 if amp_low_mid > 0.0 else -1
        sign_J_WE = 1 if amp_mid_high > 0.0 else -1

    return {
        'sign_J_NS': sign_J_NS,
        'sign_J_WE': sign_J_WE,
        'amp_low_mid': amp_low_mid,
        'amp_mid_high': amp_mid_high,
        'E_low': float(E[low_i]),
        'E_mid': float(E[mid_i]),
        'E_high': float(E[high_i]),
    }


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------

def classify_region(theta_C, theta_P, K_s=1.0):
    """
    Classify a (theta_C, theta_P) point into one of the six phase-diagram regions.

    Region boundaries:
      L1  = (1/2)*arcsin(1/3) ~  9.74 deg  (lower rim threshold)
      L2  = 90 - L1           ~ 80.26 deg  (upper rim threshold)
      Gamma_NS: cot(theta_C)*cot(theta_P) = 2  <->  J_NS = 0
      Gamma_EW: tan(theta_C)*tan(theta_P) = 2  <->  J_WE = 0

    Regions in the rim-active band (L1 < theta_P < L2):
      A: J_NS > 0, J_WE < 0  (NS spokes + diagonal rim, beta0=1)
      B: J_NS < 0, J_WE < 0  (diagonal rim only,         beta0=3)
      C: J_NS < 0, J_WE > 0  (EW spokes + diagonal rim,  beta0=1)

    Regions in the rim-inactive zones (theta_P < L1 or theta_P > L2):
      D: J_NS > 0, J_WE < 0  (NS spokes only,  beta0=3)
      E: J_NS < 0, J_WE < 0  (no active edges,  beta0=5)
      F: J_NS < 0, J_WE > 0  (EW spokes only,  beta0=3)

    Returns
    -------
    region : str in {'A', 'B', 'C', 'D', 'E', 'F'}
    J_values : dict with keys 'J_NS', 'J_WE', 'J_diag', 'J_anti'.
    """
    J_NS, J_WE, J_diag, J_anti = w4_couplings(theta_C, theta_P, K_s)
    rim_active = J_diag > 0

    if rim_active:
        if J_NS > 0:
            region = 'A'
        elif J_WE > 0:
            region = 'C'
        else:
            region = 'B'
    else:
        if J_NS > 0:
            region = 'D'
        elif J_WE > 0:
            region = 'F'
        else:
            region = 'E'

    return region, {'J_NS': J_NS, 'J_WE': J_WE, 'J_diag': J_diag, 'J_anti': J_anti}


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def lorentzian(omega, E, gamma):
    """Normalised Lorentzian centred at E with half-width gamma."""
    return (gamma / np.pi) / ((omega - E) ** 2 + gamma ** 2)


def linear_absorption(energies, mu_x, mu_y, gamma=0.05, n_omega=500, omega_range=None):
    """
    Compute the linear absorption spectrum.

    A(omega) = sum_alpha  |mu^alpha|^2 * L(omega - E_alpha)

    where L is a Lorentzian of half-width gamma.  Only bright states (nonzero
    oscillator strength) contribute.

    Parameters
    ----------
    energies : (N,) ndarray.
    mu_x, mu_y : (N,) ndarray, transition dipole components.
    gamma : float, Lorentzian half-width (same units as energies).
    n_omega : int, number of frequency points.
    omega_range : (float, float) or None.

    Returns
    -------
    omega : (n_omega,) ndarray.
    spectrum : (n_omega,) ndarray.
    """
    osc = oscillator_strengths(mu_x, mu_y)

    if omega_range is None:
        e_min, e_max = energies.min(), energies.max()
        pad = max(4.0 * gamma, 0.1 * (e_max - e_min) + 0.01)
        omega_range = (e_min - pad, e_max + pad)

    omega = np.linspace(omega_range[0], omega_range[1], n_omega)
    spectrum = np.zeros(n_omega)
    for E, f in zip(energies, osc):
        spectrum += f * lorentzian(omega, E, gamma)

    return omega, spectrum


def two_des_spectrum(energies, mu_x, mu_y, gamma=0.05, n_omega=150,
                     omega_range=None, sequence='XXXX'):
    """
    Compute the absorptive 2DES spectrum in the impulsive limit at T = 0.

    For a fixed-orientation (surface-immobilised) sample, the GSB + SE
    contribution to the absorptive 2D spectrum is:

      A(omega1, omega3) = sum_{a,b}  A_ab * L(omega1 - E_a) * L(omega3 - E_b)

    where A_ab is the polarisation-sequence-dependent amplitude:
      'XXXX':  A_ab = (mu_x^a)^2 * (mu_x^b)^2
      'YYYY':  A_ab = (mu_y^a)^2 * (mu_y^b)^2
      'XYXY':  A_ab = mu_x^a * mu_y^a * mu_x^b * mu_y^b
      'XXYY':  A_ab = (mu_x^a)^2 * (mu_y^b)^2

    Note: this model includes only the ground-state bleach (GSB) and stimulated
    emission (SE) pathways.  Excited-state absorption (ESA) from two-exciton
    states is not included.  All diagonal and cross-peak amplitudes are >= 0.

    Parameters
    ----------
    energies : (N,) ndarray.
    mu_x, mu_y : (N,) ndarray, transition dipole components.
    gamma : float, Lorentzian half-width.
    n_omega : int, number of frequency points along each axis.
    omega_range : (float, float) or None.
    sequence : str, polarisation sequence ('XXXX', 'YYYY', 'XYXY', 'XXYY').

    Returns
    -------
    omega1 : (n_omega,) ndarray  (excitation axis).
    omega3 : (n_omega,) ndarray  (detection axis).
    spectrum : (n_omega, n_omega) ndarray, A(omega1, omega3).
    """
    if omega_range is None:
        e_min, e_max = energies.min(), energies.max()
        pad = max(4.0 * gamma, 0.1 * (e_max - e_min) + 0.01)
        omega_range = (e_min - pad, e_max + pad)

    omega1 = np.linspace(omega_range[0], omega_range[1], n_omega)
    omega3 = np.linspace(omega_range[0], omega_range[1], n_omega)

    seq = sequence.upper()
    if seq == 'XXXX':
        amp_a = mu_x ** 2
        amp_b = mu_x ** 2
    elif seq == 'YYYY':
        amp_a = mu_y ** 2
        amp_b = mu_y ** 2
    elif seq == 'XYXY':
        amp_a = mu_x * mu_y
        amp_b = mu_x * mu_y
    elif seq == 'XXYY':
        amp_a = mu_x ** 2
        amp_b = mu_y ** 2
    else:
        raise ValueError(f"Unknown polarisation sequence: {sequence!r}")

    # Precompute Lorentzians for all states on each axis
    L1 = np.stack([lorentzian(omega1, E, gamma) for E in energies])  # (N, n_omega)
    L3 = np.stack([lorentzian(omega3, E, gamma) for E in energies])  # (N, n_omega)

    # Outer-product accumulation: faster than a double Python loop for typical N
    spectrum = np.einsum('a,b,ai,bj->ij', amp_a, amp_b, L1, L3)
    return omega1, omega3, spectrum


def two_des_anisotropy(energies, mu_x, mu_y, gamma=0.05, n_omega=150, omega_range=None):
    """
    Compute the 2DES polarisation anisotropy r(omega1, omega3).

      r = (A_XXXX - A_XYXY) / (A_XXXX + 2 * A_XYXY)

    This is orientation-independent for an isotropic ensemble of in-plane dipoles
    and measures the relative angle between excitation and detection transitions.

    Returns
    -------
    omega1, omega3 : (n_omega,) ndarrays.
    anisotropy : (n_omega, n_omega) ndarray.
    """
    o1, o3, A_XXXX = two_des_spectrum(energies, mu_x, mu_y, gamma, n_omega, omega_range, 'XXXX')
    _,  _,  A_XYXY = two_des_spectrum(energies, mu_x, mu_y, gamma, n_omega, omega_range, 'XYXY')

    denom = A_XXXX + 2.0 * A_XYXY
    with np.errstate(invalid='ignore', divide='ignore'):
        anisotropy = np.where(denom > 1e-30, (A_XXXX - A_XYXY) / denom, 0.0)
    return o1, o3, anisotropy


# ---------------------------------------------------------------------------
# Berry phase (adiabatic parallel transport)
# ---------------------------------------------------------------------------

def berry_phase_sign(loop_theta_C, loop_theta_P, K_s=1.0, state_idx=None,
                     n_substeps=4):
    """
    Compute the sign accumulated by symmetric-block eigenstate(s) after one
    closed loop in (theta_C, theta_P) parameter space.

    Uses parallel transport: at each step choose the sign of the new eigenvector
    such that its overlap with the previous eigenvector is positive.

    A final overlap of +1 with the initial vector indicates Berry phase 0 (mod 2pi).
    A final overlap of -1 indicates Berry phase pi (sign flip).

    Parameters
    ----------
    loop_theta_C, loop_theta_P : array-like
        Sequence of N parameter-space points.  The first point is NOT repeated
        at the end; the function closes the loop internally.
    K_s : float.
    state_idx : int or None
        Index of the symmetric-block eigenstate to track (0=lowest, 1, 2=highest).
        If None, returns a list of three signs (one per eigenstate).
    n_substeps : int
        Number of interpolation substeps between consecutive loop points.
        Increase for smoother parallel transport near the CI.

    Returns
    -------
    sign : int (+1 or -1) if state_idx is given, else list of three int.
    """
    loop_theta_C = np.asarray(loop_theta_C, dtype=float)
    loop_theta_P = np.asarray(loop_theta_P, dtype=float)
    N = len(loop_theta_C)

    # Build a finer parameter sequence by linear interpolation between loop points
    tc_fine, tp_fine = [], []
    for i in range(N):
        i_next = (i + 1) % N
        for s in range(n_substeps):
            t = s / n_substeps
            tc_fine.append((1.0 - t) * loop_theta_C[i] + t * loop_theta_C[i_next])
            tp_fine.append((1.0 - t) * loop_theta_P[i] + t * loop_theta_P[i_next])
    tc_fine = np.array(tc_fine)
    tp_fine = np.array(tp_fine)
    M_total = len(tc_fine)

    def _transport_one(k):
        M0 = symmetric_block(tc_fine[0], tp_fine[0], K_s)
        _, evecs = sp_eigh(M0)
        v_prev = evecs[:, k].copy()
        v_initial = v_prev.copy()

        for i in range(1, M_total):
            M = symmetric_block(tc_fine[i], tp_fine[i], K_s)
            _, evecs_new = sp_eigh(M)
            overlaps = evecs_new.T @ v_prev
            best = int(np.argmax(np.abs(overlaps)))
            v_curr = evecs_new[:, best].copy()
            if overlaps[best] < 0.0:
                v_curr = -v_curr
            v_prev = v_curr

        # Close the loop: transport back to the initial eigenvector
        M_close = symmetric_block(tc_fine[0], tp_fine[0], K_s)
        _, evecs_close = sp_eigh(M_close)
        overlaps_close = evecs_close.T @ v_prev
        best_close = int(np.argmax(np.abs(overlaps_close)))
        v_final = evecs_close[:, best_close].copy()
        if overlaps_close[best_close] < 0.0:
            v_final = -v_final

        return int(np.sign(np.dot(v_final, v_initial)))

    if state_idx is not None:
        return _transport_one(state_idx)
    else:
        return [_transport_one(k) for k in range(3)]


# ---------------------------------------------------------------------------
# Convenience: spectroscopic tell summary
# ---------------------------------------------------------------------------

def spectroscopic_summary(theta_C, theta_P, K_s=1.0, epsilon=0.0,
                          delta_epsilon=0.0, mu=1.0):
    """
    Return a dict of key spectroscopic properties for a given orientation.

    Useful for scanning the phase diagram.

    Returns
    -------
    dict with keys:
      'region': str
      'J_NS', 'J_WE', 'J_diag', 'J_anti': float (coupling constants)
      'eigenvalues': (5,) ndarray
      'osc_strengths': (5,) ndarray (|mu^alpha|^2)
      'n_bright': int (number of states with osc > 1e-6 * mu^2)
      'dark_state_energies': (2,) ndarray (analytic antisymmetric energies)
      'bright_energies': list of float
      'bright_osc': list of float
    """
    region, J_vals = classify_region(theta_C, theta_P, K_s)
    evals, evecs = solve_w4(theta_C, theta_P, K_s, epsilon, delta_epsilon)
    mu_x, mu_y = transition_dipoles(evecs, theta_C, theta_P, mu)
    osc = oscillator_strengths(mu_x, mu_y)

    K_r = K_s / (2.0 * np.sqrt(2.0))
    E_dark_plus  = epsilon + 3.0 * K_r * np.sin(2.0 * theta_P)
    E_dark_minus = epsilon - 3.0 * K_r * np.sin(2.0 * theta_P)

    threshold = 1e-6 * mu ** 2
    bright_mask = osc > threshold

    return {
        'region': region,
        **J_vals,
        'eigenvalues': evals,
        'osc_strengths': osc,
        'n_bright': int(bright_mask.sum()),
        'dark_state_energies': np.array([E_dark_minus, E_dark_plus]),
        'bright_energies': list(evals[bright_mask]),
        'bright_osc': list(osc[bright_mask]),
    }
