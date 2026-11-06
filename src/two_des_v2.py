from pgsc import frenkel_coupling
import numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class System:
    def __init__(self, locations, tdm_angles):
        self.locations = locations
        self.tdm_angles = tdm_angles
        self.hamiltonian = np.zeros((len(locations), len(locations)), dtype=complex)
        self.n = len(locations)
        self.epsilon = 0.0  # Initialize epsilon to zero
        self.construct_hamiltonian()

    def construct_hamiltonian(self, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        self.epsilon = epsilon
        # Construct the Hamiltonian matrix based on the locations and tdm_angles
        self.hamiltonian = np.eye(self.n, dtype=complex) * epsilon
        for i in range(self.n):
            for j in range(i):
                # Calculate the interaction term based on the locations and tdm_angles
                interaction = frenkel_coupling(self.tdm_angles[i], self.tdm_angles[j], np.atan2(self.locations[j][1] - self.locations[i][1], self.locations[j][0] - self.locations[i][0]), np.linalg.norm(np.array(self.locations[i]) - np.array(self.locations[j])))
                # Update the Hamiltonian matrix with the interaction term
                self.hamiltonian[i, j] = interaction
                self.hamiltonian[j, i] = interaction  # Ensure Hermitian property
        return self.hamiltonian.copy()
    
    def compute_eigen(self, epsilon=None):
        # Compute the eigenvalues and eigenvectors of the Hamiltonian
        self.construct_hamiltonian(epsilon)
        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
        return eigenvalues, eigenvectors

    def _two_exciton_model(self):
        """Rotating-frame Hamiltonian, dipoles and site occupations (0/1/2 excitons).

        Basis order: |g>, |0> ... |n-1>, |0,1> ... |n-2,n-1>.
        Chromophores are hard-core two-level sites: no double occupancy.
        """
        h1 = np.asarray(self.construct_hamiltonian(), dtype=complex).copy()
        h1 -= self.epsilon * np.eye(self.n)
        pairs = list(combinations(range(self.n), 2))
        dim = 1 + self.n + len(pairs)
        ham = np.zeros((dim, dim), dtype=complex)
        ham[1:1+self.n, 1:1+self.n] = h1

        # The two-exciton Hamiltonian is the hard-core two-particle lift of H1.
        for p, (i, j) in enumerate(pairs):
            a = 1 + self.n + p
            ham[a, a] = h1[i, i] + h1[j, j]
            for q in range(p):
                k, l = pairs[q]
                common = {i, j} & {k, l}
                if len(common) == 1:
                    dst = next(iter({i, j} - common))
                    src = next(iter({k, l} - common))
                    ham[a, 1+self.n+q] = h1[dst, src]
                    ham[1+self.n+q, a] = h1[src, dst]

        lowering = {}
        for polarization, dipoles in (
            ('X', np.cos(self.tdm_angles)),
            ('Y', np.sin(self.tdm_angles)),
        ):
            op = np.zeros((dim, dim), dtype=complex)
            for i, mu_i in enumerate(dipoles):
                op[0, 1+i] = mu_i
            for p, (i, j) in enumerate(pairs):
                f = 1 + self.n + p
                op[1+i, f] = dipoles[j]
                op[1+j, f] = dipoles[i]
            lowering[polarization] = op

        occupations = np.zeros((self.n, dim), dtype=float)
        for i in range(self.n):
            occupations[i, 1+i] = 1.
        for p, (i, j) in enumerate(pairs):
            occupations[i, 1+self.n+p] = 1.
            occupations[j, 1+self.n+p] = 1.

        return ham, lowering, occupations

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
            dim = 1 + self.n + self.n*(self.n-1)//2
            return np.empty((0, 2, dim, dim), dtype=complex)
        if np.any(np.diff(t1_values) < 0):
            raise ValueError('t1_values must be sorted in ascending order')
        solver_times = np.unique(np.concatenate(([0.0], t1_values)))

        ham, lowering_ops, occupations = self._two_exciton_model()
        ham_qobj = Qobj(ham)
        td_op_x = lowering_ops['X']
        td_op_y = lowering_ops['Y']
        lowering = td_op_x if sequence[0] == 'X' else td_op_y
        raising = lowering.T
        rho_0 = np.zeros(ham.shape, dtype=complex)
        rho_0[0, 0] = 1.0
        initial_states = (
            raising @ rho_0 - rho_0 @ raising,
            rho_0 @ lowering - lowering @ rho_0,
        )

        redfield_operators = []
        for site in range(self.n):
            coupling_operator = np.diag(site_coupling * occupations[site]).astype(complex)

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
        include_esa=True,
    ):
        """Compute one 2DES row.
        Exactly combines GSB, SE, and ESA by linearity of the Redfield
        propagator. GSB stays in the uncoupled ground-state population
        during t2 and therefore requires no t2 solver call.
        """
        from qutip import Qobj, brmesolve

        if eta < 0 or cutoff <= 0 or temperature < 0 or nsteps < 1:
            raise ValueError('Invalid Redfield bath or solver parameters')
        if len(sequence) != 4 or any(p not in 'XY' for p in sequence):
            raise ValueError('sequence must be four X/Y polarization labels')
        state_after_t1 = np.asarray(state_after_t1, dtype=complex)
        dim = 1 + self.n + self.n * (self.n - 1) // 2
        if state_after_t1.shape != (2, dim, dim):
            raise ValueError(f'state_after_t1 must have shape (2, {dim}, {dim})')
        t3_values = np.asarray(t3_values, dtype=float)
        if t3_values.ndim != 1 or t3_values.size == 0:
            raise ValueError('t3_values must be a nonempty one-dimensional array')
        if np.any(t3_values < 0):
            raise ValueError('t3_values must be non-negative')
        solver_times = np.unique(np.concatenate(([0.0], t3_values)))
        t3_indices = np.searchsorted(solver_times, t3_values)

        ham, lowering_ops, occupations = self._two_exciton_model()
        ham_qobj = Qobj(ham)
        lower2 = lowering_ops[sequence[1]]
        raise2 = lower2.conj().T
        raise3 = lowering_ops[sequence[2]].conj().T
        detection = lowering_ops[sequence[3]]

        def ohmic_spectrum(omega):
            omega = float(omega)
            frequency = abs(omega)
            if frequency == 0.0:
                return 0.0  # Existing bath convention; fix separately.
            spectral_density = (2.0 * np.pi * eta * frequency
                                * np.exp(-frequency / cutoff))
            if temperature == 0.0:
                return spectral_density if omega > 0.0 else 0.0
            occupation = 1.0 / np.expm1(frequency / temperature)
            return spectral_density * (occupation + 1.0 if omega > 0.0
                                       else occupation)

        redfield_operators = [
            (Qobj(np.diag(site_coupling * occupations[site])), ohmic_spectrum)
            for site in range(self.n)
        ]

        def propagate(rho, times):
            if len(times) == 1 or times[-1] == times[0]:
                return [rho]
            result = brmesolve(
                ham_qobj, Qobj(rho), times, redfield_operators,
                sec_cutoff=sec_cutoff, options={'nsteps': nsteps},
            )
            return [state.full() for state in result.states]

        nr_t1, r_t1 = state_after_t1
        # Ground-state bleaching: these are proportional to |g><g|.
        # Their populations are stationary during t2 because the ground
        # state is isolated by both the Hamiltonian and bath operators.
        nr_gsb = lower2 @ nr_t1
        r_gsb = r_t1 @ raise2

        # Stimulated emission and ESA share the SAME state during t2.
        nr_excited = propagate(nr_t1 @ lower2, [0.0, t2])[-1]
        r_excited = propagate(raise2 @ r_t1, [0.0, t2])[-1]

        # Combine all pathways BEFORE t3 evolution (linearity).
        nr_start_t3 = raise3 @ nr_gsb + nr_excited @ raise3
        r_start_t3 = raise3 @ r_gsb + r_excited @ raise3
        if include_esa:
            nr_start_t3 -= raise3 @ nr_excited
            r_start_t3 -= raise3 @ r_excited

        nr_states = propagate(nr_start_t3, solver_times)
        r_states = propagate(r_start_t3, solver_times)
        S_NR = np.array([np.trace(detection @ nr_states[i])
                         for i in t3_indices], dtype=complex)
        S_R = np.array([np.trace(detection @ r_states[i])
                        for i in t3_indices], dtype=complex)
        return S_R, S_NR

    def two_des_spectrum(
    self, t2, sequence='XXXX', n_processes=None, include_esa=True
):
        '''Compute a 2DES spectrum, evaluating time points in parallel.

        ``n_processes`` controls the process-pool size for both calculation
        modes. Redfield rows are evaluated sequentially after all ``t1``
        states have been generated.
        '''

        dt1 = 0.1
        dt3 = 0.1
        t1 = np.arange(0, 25, dt1)
        t3 = np.arange(0, 25, dt3)

        if n_processes is not None and n_processes < 1:
            raise ValueError('n_processes must be positive')

        t1_states = self.generate_t1_states(t1, sequence=sequence)

        S_R_rows = []
        S_NR_rows = []

        for state_after_t1 in t1_states:
            row_R, row_NR = self.two_des_spectrum_1row_redfield(
                state_after_t1, t2, t3, sequence=sequence,
                include_esa=include_esa
            )

            S_R_rows.append(row_R)
            S_NR_rows.append(row_NR)

        # Both arrays have shape (len(t3), len(t1)).
        S_R = np.asarray(S_R_rows, dtype=complex).T
        S_NR = np.asarray(S_NR_rows, dtype=complex).T

        n3, n1 = S_R.shape

        # Rephasing:
        # Negative Fourier exponent along t1,
        # positive Fourier exponent along t3.
        F_R = np.fft.fftshift(
            np.fft.fft(
                np.fft.ifft(S_R, axis=0) * n3,
                axis=1
            ),
            axes=(0, 1)
        )

        # Nonrephasing:
        # Positive Fourier exponent along both t1 and t3.
        F_NR = np.fft.fftshift(
            np.fft.ifft2(S_NR, axes=(0, 1)) * (n1 * n3),
            axes=(0, 1)
        )

        # Combine the two signals on the same frequency axes.
        S_fft = F_R + F_NR

        omega1 = 2 * np.pi * np.fft.fftshift(
            np.fft.fftfreq(len(t1), d=dt1)
        )

        omega3 = 2 * np.pi * np.fft.fftshift(
            np.fft.fftfreq(len(t3), d=dt3)
        )

        # Restore absolute optical frequencies.
        omega_ref = self.epsilon
        omega1 += omega_ref
        omega3 += omega_ref
        return omega1, omega3, S_fft

    def plot_2des(self, ax, t2=1.0, zoom_factor=1.0, log=False, n_processes=None):
        '''Plot the 2DES spectrum.

        The Hamiltonian and time variables use units where K_s = 1
        and hbar = 1. Since K_s represents 100 cm^-1 in this model,
        the plotted angular frequencies are normalized by K_s.
        '''

        omega1, omega3, S = self.two_des_spectrum(
            t2, n_processes=n_processes
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
        S = S[start3:end3, start1:end1]

        if log:
            # Preserve original logarithmic magnitude visualization
            S = np.log(np.abs(S) + 1e-10)
            cmap = 'viridis'
            norm = None

        else:
            S = np.real(S)

            vmax = np.max(np.abs(S))
            if vmax == 0:
                vmax = 1.0

            cmap = 'RdBu_r'

            norm = mcolors.AsinhNorm(
                linear_width=0.02 * vmax,
                vmin=-vmax,
                vmax=vmax
            )


        im = ax.imshow(
            S,
            extent=(
                omega1.min(), omega1.max(),
                omega3.min(), omega3.max()
            ),
            origin='lower',
            aspect='auto',
            cmap=cmap,
            norm=norm
        )

        ax.set_xlabel(r'$\omega_1/K_s$')
        ax.set_ylabel(r'$\omega_3/K_s$')
        ax.set_title(f'2DES Spectrum at $t_2={t2}$')

        if log:
            plt.colorbar(im, ax=ax, label='Log Signal Intensity')
        else:
            plt.colorbar(im, ax=ax, label='Signal Intensity')

    def two_des_anisotropy(self, t2, n_processes=None):
        omega1, omega3, S_xxxx = self.two_des_spectrum(
            t2, sequence='XXXX', n_processes=n_processes
        )
        _, _, S_xyxy = self.two_des_spectrum(
            t2, sequence='XYXY', n_processes=n_processes
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            anisotropy = (S_xxxx - S_xyxy) / (S_xxxx + 2 * S_xyxy)
        return omega1, omega3, anisotropy

    def plot_2des_anisotropy(
        self, ax, t2=1.0, zoom_factor=1.0, n_processes=None
    ):
        omega1, omega3, anisotropy = self.two_des_anisotropy(t2, n_processes=n_processes)

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

    def linear_absorption_spectrum(self, x=True):
        dt = 0.1
        t = np.arange(0, 120, dt)
        S = np.zeros_like(t, dtype=complex)
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
        omega_ref = self.epsilon
        ham -= omega_ref * np.eye(self.n)

        ham = np.vstack((
            np.zeros((1, self.n)), ham
        ))
        ham = np.hstack((
            np.zeros((self.n + 1, 1)), ham
        ))
        ham_qobj = Qobj(ham)

        # Transition dipoles for the selected polarization
        dipoles = np.array(
            np.cos(self.tdm_angles).tolist()
            if x else
            np.sin(self.tdm_angles).tolist(),
            dtype=complex,
        )

        # Raising and lowering operators
        raising = np.zeros(
            (self.n + 1, self.n + 1), dtype=complex
        )
        raising[1:, 0] = dipoles
        lowering = raising.conj().T

        rho_0 = np.zeros(
            (self.n + 1, self.n + 1), dtype=complex
        )
        rho_0[0, 0] = 1.0

        # Only the absorption coherence
        rho_after_pulse = raising @ rho_0

        # Redfield bath operators
        redfield_operators = []
        for site in range(1, self.n + 1):
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
        solver_times = np.append(t, 120.0)

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
        omega += self.epsilon

        return omega, S_fft

    def plot_linear_absorption(
        self, ax, x=True, zoom_factor=1.0
    ):
        omega, S = self.linear_absorption_spectrum(x=x)

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



