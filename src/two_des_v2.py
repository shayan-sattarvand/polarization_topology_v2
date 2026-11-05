from pgsc import frenkel_coupling
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _init_two_des_worker(wheel):
    global _TWO_DES_WORKER_WHEEL
    _TWO_DES_WORKER_WHEEL = wheel


def _two_des_point_worker(args):
    t1, t2, t3, sequence, redfield = args
    return _TWO_DES_WORKER_WHEEL.two_des_spectrum_1pt(
        t1, t2, t3, sequence=sequence
    )



class System:
    def __init__(self, locations, tdm_angles):
        self.locations = locations
        self.tdm_angles = tdm_angles
        self.hamiltonian = np.zeros((len(locations), len(locations)), dtype=complex)
        self.construct_hamiltonian()
        self.n = len(locations)

    def construct_hamiltonian(self, epsilon=0.0):
        # Construct the Hamiltonian matrix based on the locations and tdm_angles
        for i in range(self.n):
            for j in range(i):
                # Calculate the interaction term based on the locations and tdm_angles
                interaction = frenkel_coupling(self.tdm_angles[i], self.tdm_angles[j], np.atan2(self.locations[j][1] - self.locations[i][1], self.locations[j][0] - self.locations[i][0]), np.linalg.norm(np.array(self.locations[i]) - np.array(self.locations[j])))
                # Update the Hamiltonian matrix with the interaction term
                self.hamiltonian[i, j] = interaction
                self.hamiltonian[j, i] = interaction  # Ensure Hermitian property
    
    def compute_eigen(self, epsilon=0.0):
        # Compute the eigenvalues and eigenvectors of the Hamiltonian
        self.construct_hamiltonian(epsilon)
        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
        return eigenvalues, eigenvectors

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
            return np.empty((0, 2, self.n + 1, self.n + 1), dtype=complex)
        if np.any(np.diff(t1_values) < 0):
            raise ValueError('t1_values must be sorted in ascending order')
        solver_times = np.unique(np.concatenate(([0.0], t1_values)))

        ham = self.construct_hamiltonian()
        ham -= 150.0 * np.eye(self.n)
        ham = np.vstack((np.zeros((1, self.n)), ham))
        ham = np.hstack((np.zeros((self.n + 1, 1)), ham))
        ham_qobj = Qobj(ham)

        td_op_x = np.array([0] + np.cos(self.tdm_angles).tolist(), dtype=complex)
        td_op_x = np.vstack((
            td_op_x,
            np.zeros((self.n, self.n + 1), dtype=complex),
        ))
        td_op_y = np.array([0] + np.sin(self.tdm_angles).tolist(), dtype=complex)
        td_op_y = np.vstack((
            td_op_y,
            np.zeros((self.n, self.n + 1), dtype=complex),
        ))
        lowering = td_op_x if sequence[0] == 'X' else td_op_y
        raising = lowering.T
        rho_0 = np.zeros((self.n + 1, self.n + 1), dtype=complex)
        rho_0[0, 0] = 1.0
        initial_states = (
            raising @ rho_0 - rho_0 @ raising,
            rho_0 @ lowering - lowering @ rho_0,
        )

        redfield_operators = []
        for site in range(1, self.n + 1):
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
        ham -= 150.0 * np.eye(self.n)
        ham = np.vstack((np.zeros((1, self.n)), ham))
        ham = np.hstack((np.zeros((self.n + 1, 1)), ham))
        ham_qobj = Qobj(ham)

        td_op_x = np.array([0] + np.cos(self.tdm_angles).tolist(), dtype=complex)
        td_op_x = np.vstack((
            td_op_x,
            np.zeros((self.n, self.n + 1), dtype=complex),
        ))
        td_op_y = np.array([0] + np.sin(self.tdm_angles).tolist(), dtype=complex)
        td_op_y = np.vstack((
            td_op_y,
            np.zeros((self.n, self.n + 1), dtype=complex),
        ))
        td_op_lowering_x = td_op_x
        td_op_raising_x = td_op_x.T
        td_op_lowering_y = td_op_y
        td_op_raising_y = td_op_y.T

        # Site-energy fluctuations do not couple the ground state.  A
        # non-secular Redfield treatment is requested with sec_cutoff=-1.
        redfield_operators = []
        for site in range(1, self.n + 1):
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
            # return operation_naive(rho, left, raising, x) - operation_naive(
            #     rho, not left, raising, x
            # ) USING A COMMUTATOR IS WRONG HERE
            return operation_naive(rho, left, raising, x)

        signal_operator = td_op_x if sequence[3] == 'X' else td_op_y
        
        pathway_specs = (
            # Nonrephasing GSB
            (state_after_t1[0], True, False, True, True),

            # Rephasing GSB
            (state_after_t1[1], False, True, True, True),

            # Nonrephasing SE
            (state_after_t1[0], False, False, False, True),

            # Rephasing SE
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

        rho_NR = [
            pathways[0][i] + pathways[2][i]
            for i in range(t3_values.size)
        ]

        rho_R = [
            pathways[1][i] + pathways[3][i]
            for i in range(t3_values.size)
        ]

        S_NR = np.asarray([
            np.trace(signal_operator @ rho)
            for rho in rho_NR
        ])

        S_R = np.asarray([
            np.trace(signal_operator @ rho)
            for rho in rho_R
        ])

        return S_R, S_NR

    def two_des_spectrum(
    self, t2, sequence='XXXX', n_processes=None
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
                state_after_t1, t2, t3, sequence=sequence
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
        omega_ref = 150.0
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




