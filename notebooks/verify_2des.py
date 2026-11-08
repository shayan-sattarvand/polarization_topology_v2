"""Independent analytic (sum-over-eigenstates) benchmarks for System.two_des_spectrum_batched.

Run next to two_des_v2.py and pgsc.py:
    python verify_2des_analytic.py --source two_des_v2.py

Tests several exactly solvable, CLOSED (eta=0) Frenkel aggregates. The reference
uses analytic eigenstate pathway sums and finite geometric Fourier sums, not
Redfield propagation or FFT. No plots; a nonzero exit code means failure.

By default it runs both an exact unitary-generator test (no QuTiP needed) and,
when QuTiP is installed, a second end-to-end test of QuTiP's eta=0 Redfield
construction. Bath-induced dissipation is NOT covered by this benchmark.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np


def load_system_class(path):
    path = Path(path).expanduser().resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('user_2des_module_for_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.System


def make_system_class(Base, generator_mode):
    class TestSystem(Base):
        def __init__(self, h1, angles, epsilon=150.0):
            h1 = np.asarray(h1, dtype=complex)
            assert h1.ndim == 2 and h1.shape[0] == h1.shape[1]
            np.testing.assert_allclose(h1, h1.conj().T, atol=1e-13)
            self.n = h1.shape[0]
            self.locations = np.column_stack((np.arange(self.n), np.zeros(self.n)))
            self.tdm_angles = np.asarray(angles, dtype=float)
            self.epsilon = float(epsilon)
            self.hamiltonian = h1 + epsilon * np.eye(self.n)
            self._redfield_cache_key = None
            self._redfield_cache_tensor = None

        def construct_hamiltonian(self, epsilon=None):
            if epsilon is not None:
                self.hamiltonian += (float(epsilon) - self.epsilon) * np.eye(self.n)
                self.epsilon = float(epsilon)
            return self.hamiltonian.copy()

        def _get_redfield_tensor(self, ham, occupations, *args, **kwargs):
            if generator_mode == 'unitary':
                dim = ham.shape[0]
                # vec(-i[H,rho]) = -i (I kron H - H^T kron I) vec(rho).
                generator = -1j * (np.kron(np.eye(dim), ham)
                                   - np.kron(ham.T, np.eye(dim)))
                return SimpleNamespace(full=lambda: generator)
            # Force an exactly closed system while retaining real QuTiP
            # Redfield-tensor construction and all other user code.
            kwargs['eta'] = 0.0
            return super()._get_redfield_tensor(ham, occupations, *args, **kwargs)
    return TestSystem


def geometric(delta, n, dt):
    """Exact dt * sum_{k=0}^{n-1} exp(i*delta*k*dt), including removable poles."""
    # Reduce modulo 2*pi so Fourier aliases have their correct removable limit.
    phase = (np.asarray(delta, dtype=float) * dt + np.pi) % (2*np.pi) - np.pi
    half = 0.5 * phase
    # sinc formulation is stable near delta=0, and also handles near aliases.
    denominator = np.sinc(half / np.pi)
    numerator = np.sinc(n * half / np.pi)
    ratio = np.divide(numerator, denominator, out=np.ones_like(numerator),
                      where=np.abs(denominator) > 1e-12)
    return n * dt * np.exp(1j * (n - 1) * half) * ratio


def analytic_spectrum(system, t2, sequence, include_esa, dt=0.3, tmax=25.0):
    """Closed-form GSB + SE - ESA in exciton eigenstates, then geometric sums.

    No Liouvillian, time stepping, or FFT is used. This reference follows
    the same phase-matching convention as the user's six-pathway calculation.
    """
    h, lowering, _ = system._two_exciton_model()
    n = system.n
    one = slice(1, 1+n)
    two = slice(1+n, h.shape[0])
    E, V = np.linalg.eigh(h[one, one])
    F, W = np.linalg.eigh(h[two, two])
    d = {}
    m = {}
    for p in 'XY':
        dip = lowering[p]
        d[p] = dip[0, one] @ V
        m[p] = V.conj().T @ dip[one, two] @ W

    n1 = len(np.arange(0, tmax, dt))
    n3 = n1
    omega1 = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n1, dt))
    omega3 = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n3, dt))
    # Arrays indexed [exciton index, frequency index].
    g1nr = geometric(omega1[None, :] - E[:, None], n1, dt)
    g1r = geometric(E[:, None] - omega1[None, :], n1, dt)
    g3 = geometric(omega3[None, :] - E[:, None], n3, dt)
    # [single index b, double index u, frequency index]
    g3esa = geometric(omega3[None, None, :] -
                      (F[None, :, None] - E[:, None, None]), n3, dt)

    p1, p2, p3, p4 = sequence
    d1, d2, d3, d4 = (d[p] for p in sequence)
    m3, m4 = m[p3], m[p4]
    S_nr = np.zeros((n3, n1), dtype=complex)
    S_r = np.zeros_like(S_nr)

    # Ground-state bleach: one excitation resonance during t1, another
    # during t3; the ground population is stationary during t2.
    det = np.einsum('c,c,cl->l', d3.conj(), d4, g3)
    first_nr = np.einsum('a,a,ak->k', d1.conj(), d2, g1nr)
    first_r = np.einsum('a,a,ak->k', d1, d2.conj(), g1r)
    S_nr += np.outer(det, first_nr)
    S_r += np.outer(det, first_r)

    for a in range(n):
        for b in range(n):
            # rho_ab(t2) carries exp[-i(E_a-E_b)t2].
            waiting = np.exp(-1j * (E[a] - E[b]) * t2)
            # Stimulated emission (1->0 optical coherence during t3).
            S_nr += (d1[a].conj() * d2[b] * d3[b].conj() * d4[a]
                     * waiting * np.outer(g3[a], g1nr[a]))
            S_r += (d2[a].conj() * d1[b] * d3[b].conj() * d4[a]
                    * waiting * np.outer(g3[a], g1r[b]))
            if include_esa:
                # Excited-state absorption (2->1 coherence during t3).
                for u in range(len(F)):
                    detection = m3[a, u].conj() * m4[b, u]
                    S_nr -= (d1[a].conj() * d2[b] * detection * waiting
                             * np.outer(g3esa[b, u], g1nr[a]))
                    S_r -= (d2[a].conj() * d1[b] * detection * waiting
                            * np.outer(g3esa[b, u], g1r[b]))

    return (omega1 + system.epsilon, omega3 + system.epsilon,
            S_r + S_nr, S_r, S_nr)


def check_case(system, label, t2, sequence, esa, tol):
    w1, w3, actual = system.two_des_spectrum_batched(
        t2, sequence=sequence, include_esa=esa)
    ew1, ew3, expected, _, _ = analytic_spectrum(system, t2, sequence, esa)
    np.testing.assert_allclose(w1, ew1, rtol=0, atol=1e-11)
    np.testing.assert_allclose(w3, ew3, rtol=0, atol=1e-11)
    scale = max(float(np.max(np.abs(expected))), 1e-14)
    max_error = float(np.max(np.abs(actual - expected)) / scale)
    fro_error = float(np.linalg.norm(actual - expected) /
                      max(np.linalg.norm(expected), 1e-14))
    passed = max_error < tol and fro_error < tol
    print(f"{'PASS' if passed else 'FAIL'} {label:21s} "
          f"t2={t2:4.1f} seq={sequence} ESA={str(esa):5s} "
          f"max/peak={max_error:.2e} fro={fro_error:.2e}")
    return passed


def check_generator(system, label):
    """For eta=0, QuTiP's Redfield tensor must equal -i[H,.]."""
    ham, _, occupations = system._two_exciton_model()
    R = system._get_redfield_tensor(ham, occupations).full()
    d = ham.shape[0]
    exact = -1j*(np.kron(np.eye(d), ham) - np.kron(ham.T, np.eye(d)))
    error = np.max(np.abs(R-exact))
    passed = error < 1e-10
    print(f"{'PASS' if passed else 'FAIL'} {label:21s} "
          f'eta=0 generator max error={error:.2e}')
    return passed


def run(source, tol, mode):
    Base = load_system_class(source)
    # Rotating-frame single-exciton Hamiltonians. All are exactly solvable.
    cases = [
        ('monomer', [[0.65]], [0.37],
         [(0.0, 'XXXX', False), (1.4, 'XYXY', False)]),
        ('uncoupled dimer', [[-0.7, 0.0], [0.0, 1.1]], [0.2, 1.25],
         [(0.0, 'XXXX', False), (1.0, 'XXYY', True),
          (2.0, 'XYXY', True)]),
        ('coupled dimer', [[-0.5, 0.85], [0.85, 0.9]], [0.3, 1.4],
         [(0.0, 'XXXX', True), (1.3, 'XXYY', True),
          (2.1, 'YXYX', False), (0.7, 'XYYX', True)]),
        ('asymmetric trimer', [[-0.7, 0.42, -0.21],
                               [0.42, 0.35, 0.63],
                               [-0.21, 0.63, 1.1]], [0.1, 1.1, 2.5],
         [(0.0, 'YYYY', True), (0.9, 'YYXX', True),
          (1.8, 'YXXY', True), (0.4, 'XYXY', False)]),
        ('five-site wheel', [[0.0, 0.7, -0.3, 0.4, -0.6],
                             [0.7, 0.3, 0.2, -0.1, 0.15],
                             [-0.3, 0.2, -0.5, 0.6, 0.05],
                             [0.4, -0.1, 0.6, 0.8, -0.25],
                             [-0.6, 0.15, 0.05, -0.25, -0.2]],
         [0.0, 0.5, 1.0, 1.8, 2.4],
         [(0.0, 'XXXX', True), (1.1, 'XXYY', True)]),
        ('collinear trimer', [[-0.4, 0.35, 0.1],
                              [0.35, 0.6, -0.5],
                              [0.1, -0.5, 0.2]], [0.6]*3,
         [(0.0, 'XXXX', True), (1.2, 'XXYY', True)]),
    ]
    modes = (['unitary', 'qutip'] if mode == 'both' else [mode])
    all_pass = True
    for current_mode in modes:
        if current_mode == 'qutip':
            try:
                import qutip  # noqa: F401
            except ImportError:
                print('\nSKIP QuTiP generator checks (qutip is not installed).')
                continue
        print(f'\n=== {current_mode.upper()} GENERATOR ===')
        TestSystem = make_system_class(Base, current_mode)
        for label, h1, angles, trials in cases:
            system = TestSystem(h1, angles)
            all_pass &= check_generator(system, label)
            for t2, seq, esa in trials:
                try:
                    passed = check_case(system, label, t2, seq, esa, tol)
                except Exception as exc:
                    print(f'ERROR {label} {seq} ESA={esa}: {type(exc).__name__}: {exc}')
                    passed = False
                all_pass &= passed
    print('\n' + ('ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'))
    return all_pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='two_des_v2.py',
                        help='Path to the file defining your System class')
    parser.add_argument('--tol', type=float, default=2e-9,
                        help='Maximum normalized spectral discrepancy')
    parser.add_argument('--mode', choices=('both', 'unitary', 'qutip'),
                        default='both')
    args = parser.parse_args()
    sys.exit(0 if run(args.source, args.tol, args.mode) else 1)
