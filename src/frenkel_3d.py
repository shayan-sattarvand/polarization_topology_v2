"""Three-dimensional Frenkel coupling and PGSC builders.

This module extends the 2D PGSC framework in ``src/pgsc.py`` to full 3D dipole
orientations and 3D chromophore coordinates.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pgsc import _count_components


def frenkel_coupling_3d(mu_i, mu_j, r_vec, mu=1.0):
    """Compute the 3D Frenkel coupling for a dipole pair.

    Parameters
    ----------
    mu_i : array-like, shape (3,)
        Unit vector for dipole i. Must satisfy ``|mu_i| = 1`` within 1e-10.
    mu_j : array-like, shape (3,)
        Unit vector for dipole j. Must satisfy ``|mu_j| = 1`` within 1e-10.
    r_vec : array-like, shape (3,)
        Displacement vector from i to j.
    mu : float, default=1.0
        Dipole magnitude (energy prefactor scale).

    Returns
    -------
    float
        Frenkel coupling value

        ``J = (mu^2 / |r|^3) * [(mu_i·mu_j) - 3(mu_i·r_hat)(mu_j·r_hat)]``.

    Raises
    ------
    ValueError
        If ``mu_i`` or ``mu_j`` is not unit length within 1e-10, or if
        ``|r_vec| < 1e-14``.
    """
    mu_i = np.asarray(mu_i, dtype=float)
    mu_j = np.asarray(mu_j, dtype=float)
    r_vec = np.asarray(r_vec, dtype=float)

    if mu_i.shape != (3,):
        raise ValueError(f"mu_i must have shape (3,), got {mu_i.shape}")
    if mu_j.shape != (3,):
        raise ValueError(f"mu_j must have shape (3,), got {mu_j.shape}")
    if r_vec.shape != (3,):
        raise ValueError(f"r_vec must have shape (3,), got {r_vec.shape}")

    norm_i = np.linalg.norm(mu_i)
    norm_j = np.linalg.norm(mu_j)
    if abs(norm_i - 1.0) > 1e-10:
        raise ValueError(
            f"mu_i must be a unit vector within 1e-10, got |mu_i|={norm_i:.16e}"
        )
    if abs(norm_j - 1.0) > 1e-10:
        raise ValueError(
            f"mu_j must be a unit vector within 1e-10, got |mu_j|={norm_j:.16e}"
        )

    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1e-14:
        raise ValueError(f"|r_vec| must be >= 1e-14, got {r_norm:.16e}")

    r_hat = r_vec / r_norm
    dot_ij = np.dot(mu_i, mu_j)
    dot_ir = np.dot(mu_i, r_hat)
    dot_jr = np.dot(mu_j, r_hat)

    return (mu ** 2 / r_norm ** 3) * (dot_ij - 3.0 * dot_ir * dot_jr)


def build_pgsc_3d(positions, orientations, r_cut=None, J_cut=0.0, mu=1.0):
    """Build the 3D Polarization-Gated Simplicial Complex.

    Edge ``(i, j)`` is included iff ``r_ij <= r_cut`` (or ``r_cut is None``)
    and ``J_ij > J_cut`` where ``J_ij`` is computed by ``frenkel_coupling_3d``.

    Parameters
    ----------
    positions : (N, 3) array-like
        Chromophore ``(x, y, z)`` positions.
    orientations : (N, 3) array-like
        Unit dipole vectors. Each row must be unit length.
    r_cut : float or None, default=None
        Maximum internuclear distance.
    J_cut : float, default=0.0
        Coupling threshold (edges need ``J > J_cut``).
    mu : float, default=1.0
        Dipole magnitude.

    Returns
    -------
    edges : list of (int, int)
        Active edge pairs with ``i < j``.
    beta0 : int
        Number of connected components.
    beta1 : int
        Number of independent cycles, ``beta1 = |E| - |V| + beta0``.

    Raises
    ------
    ValueError
        If ``positions`` or ``orientations`` do not have shape ``(N, 3)``, or
        if any orientation is not unit length within 1e-10.
    """
    positions = np.asarray(positions, dtype=float)
    orientations = np.asarray(orientations, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (N, 3), got {positions.shape}"
        )
    if orientations.ndim != 2 or orientations.shape[1] != 3:
        raise ValueError(
            f"orientations must have shape (N, 3), got {orientations.shape}"
        )
    if positions.shape[0] != orientations.shape[0]:
        raise ValueError(
            "positions and orientations must have the same number of rows, "
            f"got {positions.shape[0]} and {orientations.shape[0]}"
        )

    norms = np.linalg.norm(orientations, axis=1)
    bad = np.where(np.abs(norms - 1.0) > 1e-10)[0]
    if bad.size:
        k = int(bad[0])
        raise ValueError(
            "orientations rows must be unit vectors within 1e-10; "
            f"row {k} has norm {norms[k]:.16e}"
        )

    n_nodes = positions.shape[0]
    edges = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            r_vec = positions[j] - positions[i]
            r_ij = np.linalg.norm(r_vec)
            if r_cut is not None and r_ij > r_cut + 1e-12:
                continue

            J = frenkel_coupling_3d(orientations[i], orientations[j], r_vec, mu=mu)
            if J > J_cut:
                edges.append((i, j))

    beta0 = _count_components(n_nodes, edges)
    beta1 = len(edges) - n_nodes + beta0
    return edges, beta0, beta1


def w6_positions(d=1.0):
    """Return canonical node positions for the W6 octahedral tile.

    Parameters
    ----------
    d : float, default=1.0
        Octahedron circumradius (center-to-rim spoke length).

    Returns
    -------
    ndarray, shape (7, 3)
        Node positions in canonical order:

        - ``0``: ``C`` center, ``(0, 0, 0)``
        - ``1``: ``N`` north, ``(0, +d, 0)``
        - ``2``: ``E`` east, ``(+d, 0, 0)``
        - ``3``: ``S`` south, ``(0, -d, 0)``
        - ``4``: ``W`` west, ``(-d, 0, 0)``
        - ``5``: ``U`` up, ``(0, 0, +d)``
        - ``6``: ``D`` down, ``(0, 0, -d)``
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],   # C
            [0.0, d, 0.0],     # N
            [d, 0.0, 0.0],     # E
            [0.0, -d, 0.0],    # S
            [-d, 0.0, 0.0],    # W
            [0.0, 0.0, d],     # U
            [0.0, 0.0, -d],    # D
        ]
    )


def w6_pgsc(theta_C, phi_C, theta_P, phi_P, K_s=1.0, J_cut=0.0):
    """Compute the W6 PGSC using the generic 3D builder.

    Parameters
    ----------
    theta_C, phi_C : float
        Polar and azimuthal angles (radians) for the center dipole.
    theta_P, phi_P : float
        Polar and azimuthal angles (radians) for all six rim dipoles
        (uniform texture).
    K_s : float, default=1.0
        Coupling scale ``mu^2 / d^3`` with ``d = 1``.
    J_cut : float, default=0.0
        Edge threshold.

    Returns
    -------
    edges : list of (int, int)
        Active edge pairs.
    beta0 : int
        Number of connected components.
    beta1 : int
        Number of independent cycles.
    """

    def _sph_to_cart(theta, phi):
        return np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

    d = 1.0
    positions = w6_positions(d=d)

    mu_center = _sph_to_cart(theta_C, phi_C)
    mu_rim = _sph_to_cart(theta_P, phi_P)

    orientations = np.vstack([mu_center, np.tile(mu_rim, (6, 1))])

    r_cut = d * np.sqrt(2.0) * 1.001
    mu = np.sqrt(K_s)

    return build_pgsc_3d(positions, orientations, r_cut=r_cut, J_cut=J_cut, mu=mu)


def w6_betti(theta_C, phi_C, theta_P, phi_P, K_s=1.0):
    """Return ``(beta0, beta1)`` for a single W6 tile."""
    _, beta0, beta1 = w6_pgsc(theta_C, phi_C, theta_P, phi_P, K_s=K_s)
    return beta0, beta1
