import numpy as np
from .equation.equation_base import EqnBase
from .limiter import Limiter
from .flux import Flux


class Reconstruction:
    """
    Handles reconstruction methods for finite volume schemes in 1D.

    Supports piecewise constant and MUSCL methods; extensible to PPM and WENO5.
    Maintains ghost cells in reconstructed states.

    Args:
        eqn_obj (EquationBase): System for state conversions.
        str_domain (str): Reconstruction in primitive or conservative.
        str_flux (str): Flux method ('lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe').
        str_limiter (str, optional): Slope limiter type ('minmod', 'superbee', 'vanleer', etc.).
        limiter_beta (float, optional): Sharpness parameter for Osher/Sweby limiters.
    """

    def __init__(
        self,
        eqn_obj: EqnBase,
        str_reconst: str = "constant",
        str_domain: str = "primitive",
        str_flux: str = "hllc",
        str_limiter: str = "",
        limiter_beta: float = 1.5,
    ):
        if not isinstance(eqn_obj, EqnBase):
            raise TypeError("eqn_obj must be an EquationBase instance")

        self.eqn_obj = eqn_obj
        self.name = str_reconst
        self.in_primitive_domain = str_domain == "primitive"
        self.limiter_obj = (
            Limiter(str_limiter, beta=limiter_beta) if str_limiter else None
        )
        self.flux_obj = Flux(eqn_obj, str_flux, lambda_max=1.0)

        self.n_vars = self.eqn_obj.num_vars

        self.reconst_dicts = {
            "constant": self.PIECEWISE_CONSTANT,  # Lax-Friedrichs
            "muscl": self.MUSCL,
            # "ppm": self.PPM,
            # "weno": self.weno,
        }

        if self.name not in self.reconst_dicts:
            raise ValueError(
                f"Unsupported flux type: {self.name}. Choose from {list(self.reconst_dicts.keys())}"
            )

    def reconst_func(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Return the specified flux method.

        Args:
            U (np.ndarray): Conservative variables, shape (num_vars, n_cells + 2*n_ghost, ...).
            dx (np.ndarray): Spatial grid distance array.

        Returns:
            res (np.ndarray): residual flux at the interface
        """
        return self.reconst_dicts[self.name](U, dx)

    def compute_slopes(self, Q: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Compute slopes for all reconstruction methods.

        Args:
            Q (np.ndarray): Either primitive or conservative variables, shape (num_vars, n_cells_total, ...).
            dx (np.ndarray): Spatial grid distance array.

        Returns:
            dQ: (np.ndarray), slope of the left/right slopes for each cell
        """
        N = Q.shape[1]
        dQ = np.zeros((self.n_vars, N))

        # skip the 1st/last cell: 0, N-1
        for c in range(1, N - 1):
            distL = (dx[c] + dx[c - 1]) / 2
            distR = (dx[c] + dx[c + 1]) / 2
            distC = (dx[c - 1] + dx[c + 1]) / 2 + dx[c]

            dqL = (Q[:, c] - Q[:, c - 1]) / distL
            dqR = (Q[:, c + 1] - Q[:, c]) / distR
            dqC = (Q[:, c + 1] - Q[:, c - 1]) / distC

            if self.limiter_obj is not None:
                dQ[:, c] = self.limiter_obj.limiter_func(dqL, dqR, dqC)
            else:
                dQ[:, c] = dqC  # No limiting if limiter_obj is None

        return dQ

    # ------------------------------------ #
    # compute and limit slopes for all cells
    # ------------------------------------ #

    def PIECEWISE_CONSTANT(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Piecewise constant reconstruction.

        Args:
            U (np.ndarray): Conservative variables, shape (num_vars, n_cells + 2*n_ghost, ...).
            dx (np.ndarray): Spatial grid distance array.

        Returns:
            res (np.ndarray): residual flux at the interface
        """
        N = U.shape[1]
        Flux = np.zeros((self.n_vars, N - 1))
        res = np.zeros((self.n_vars, N))

        # it does not matter to constant whether to reconstruct in primitive or conservative domain

        # iterate through 1 to N-3, then separately deal with left/right edges
        #   c - cell id, i - interface id
        for i in range(1, N - 2):
            c = i
            U_L = U[:, c]
            U_R = U[:, c + 1]
            Flux[:, i] = self.flux_obj.flux_func(U_L, U_R)

            # Compute fluxes: assumes U_L and U_R are defined as the left and right states at each interface
            res[:, c] = res[:, c] + Flux[:, i] / dx[c]
            res[:, c + 1] = res[:, c + 1] - Flux[:, i] / dx[c + 1]

        # deal with leftmost face 0
        U_L = U[:, 0]
        U_R = U[:, 1]
        Flux[:, 0] = self.flux_obj.flux_func(U_L, U_R)
        res[:, 1] = res[:, 1] - Flux[:, 0] / dx[0]

        # deal with rightmost face N-2
        U_L = U[:, N - 2]
        U_R = U[:, N - 1]
        Flux[:, N - 2] = self.flux_obj.flux_func(U_L, U_R)
        res[:, N - 2] = res[:, N - 2] + Flux[:, N - 2] / dx[N - 2]

        return res

    def MUSCL(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        MUSCL reconstruction with slope limiting.

        Args:
            U (np.ndarray): Conservative variables, shape (num_vars, n_cells + 2*n_ghost, ...).
            dx (np.ndarray): Spatial grid distance array.
            n_ghost (int): Number of ghost cells per side.

        Returns:
            np.ndarray: (U_L, U_R), left and right states, each shape (num_vars, n_cells + 2*n_ghost - 1, ...).
        """
        N = U.shape[1]
        Flux = np.zeros((self.n_vars, N - 1))
        res = np.zeros((self.n_vars, N))

        if self.in_primitive_domain:
            W = self.eqn_obj.to_primitive_batch(U)
            slopes = self.compute_slopes(W, dx)

            # iterate through 1 to N-3, then separately deal with left/right edges
            #   c - cell id, i - interface id
            for i in range(1, N - 2):
                c = i
                W_L = W[:, c] + slopes[:, c] * dx[c] / 2.0
                W_R = W[:, c + 1] - slopes[:, c + 1] * dx[c + 1] / 2.0
                U_L = self.eqn_obj.to_conservative(W_L)
                U_R = self.eqn_obj.to_conservative(W_R)
                Flux[:, i] = self.flux_obj.flux_func(U_L, U_R)

                # Compute fluxes: assumes U_L and U_R are defined as the left and right states at each interface
                res[:, c] = res[:, c] + Flux[:, i] / dx[c]
                res[:, c + 1] = res[:, c + 1] - Flux[:, i] / dx[c + 1]

            # deal with leftmost face 0
            W_R = W[:, 1] - slopes[:, 1] * dx[1] / 2.0
            W_L = W_R
            U_L = self.eqn_obj.to_conservative(W_L)
            U_R = self.eqn_obj.to_conservative(W_R)
            Flux[:, 0] = self.flux_obj.flux_func(U_L, U_R)
            res[:, 1] = res[:, 1] - Flux[:, 0] / dx[0]

            # deal with rightmost face N-2
            W_L = W[:, N - 2] + slopes[:, N - 2] * dx[N - 2] / 2.0
            W_R = W_L
            U_L = self.eqn_obj.to_conservative(W_L)
            U_R = self.eqn_obj.to_conservative(W_R)
            Flux[:, N - 2] = self.flux_obj.flux_func(U_L, U_R)
            res[:, N - 2] = res[:, N - 2] + Flux[:, N - 2] / dx[N - 2]

        else:
            slopes = self.compute_slopes(U, dx)

            # iterate through 1 to N-3, then separately deal with left/right edges
            #   c - cell id, i - interface id
            for i in range(1, N - 2):
                c = i
                U_L = U[:, c] + slopes[:, c] * dx[c] / 2.0
                U_R = U[:, c + 1] - slopes[:, c + 1] * dx[c + 1] / 2.0
                Flux[:, i] = self.flux_obj.flux_func(U_L, U_R)

                # Compute fluxes: assumes U_L and U_R are defined as the left and right states at each interface
                res[:, c] = res[:, c] + Flux[:, i] / dx[c]
                res[:, c + 1] = res[:, c + 1] - Flux[:, i] / dx[c + 1]

            # deal with leftmost face 0
            U_R = U[:, 0] - slopes[:, 1] * dx[1] / 2.0
            U_L = U_R
            Flux[:, 0] = self.flux_obj.flux_func(U_L, U_R)
            res[:, 1] = res[:, 1] - Flux[:, 0] / dx[0]

            # deal with rightmost face N-2
            U_L = U[:, N - 2] + slopes[:, N - 2] * dx[N - 2] / 2.0
            U_R = U_L
            Flux[:, N - 2] = self.flux_obj.flux_func(U_L, U_R)
            res[:, N - 2] = res[:, N - 2] + Flux[:, N - 2] / dx[N - 2]

        return res

    def PPM(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Piecewise Parabolic Method (PPM) reconstruction (conservative variables).
          Constructs parabolic profiles in each cell, applies monotonicity constraints and slope limiting.
          Third-order accurate in smooth regions with full stencil, robust near discontinuities.

        Args:
            U (np.ndarray): Conservative variables, shape (num_vars, n_cells + 2*n_ghost, ...).
            dx (np.ndarray): Spatial grid distance array.
            imod_delta (bool): whether to apply the modified delta method (default: False).

        Returns:
            np.ndarray: (U_L, U_R), left and right states, each shape (num_vars, n_cells + 2*n_ghost - 1, ...).

        Raises:
            ValueError: If n_ghost is insufficient or dx <= 0.
        """
        if dx.ndim == 1:
            # n_cells_total includes the actual cells plus ghost cells
            # n_delta = n_cells_total - 2
            # n_interface = n_delta - 1

            n_vars, n_cells_total = U.shape
            n_cells_cal = n_cells_total - 4

            U_L = np.zeros((n_vars, n_cells_cal + 1))
            U_R = np.zeros((n_vars, n_cells_cal + 1))

            for j in range(n_vars):
                # Compute delta_m_A and delta_A, first 1 and last 1 cells are not used
                delta_A = np.full((n_cells_total), np.nan)
                delta_m_A = np.full((n_cells_total), np.nan)

                for i in range(1, n_cells_total - 1):
                    if (U[j, i + 1] - U[j, i]) * (U[j, i] - U[j, i - 1]) > 0:
                        gradL = 2.0 * abs(U[j, i + 1] - U[j, i])
                        gradR = 2.0 * abs(U[j, i] - U[j, i - 1])
                        delta_A[i] = (U[j, i + 1] - U[j, i - 1]) / 2.0
                        delta_m_A[i] = min(abs(delta_A[i]), gradL, gradR) * np.sign(
                            delta_A[i]
                        )
                    else:
                        delta_A[i] = (U[j, i + 1] - U[j, i - 1]) / 2.0
                        delta_m_A[i] = 0.0

                # store interface states
                E = np.full((n_cells_total - 1), np.nan)

                if imod_delta:
                    # Note: the following formula is a modification of the original PPM to include delta_m_A
                    for i in range(1, n_cells_total - 2):
                        E[i] = (
                            U[j, i]
                            + 0.5 * (U[j, i + 1] - U[j, i])
                            + (1 / 6) * (delta_m_A[i] - delta_m_A[i + 1])
                        )
                else:
                    # compute interface states using 4-point stencil
                    for i in range(1, n_cells_total - 2):
                        E[i] = (7 / 12) * (U[j, i] + U[j, i + 1]) - (1 / 12) * (
                            U[j, i - 1] + U[j, i + 2]
                        )

                # Monotonicity constraint (limit overshoots)
                for i in range(2, n_cells_total - 2):
                    aj = U[j, i]
                    aL = E[i - 1]
                    aR = E[i]
                    a_avg = 0.5 * (aL + aR)
                    a_delta = aR - aL

                    if (aR - aj) * (aj - aL) <= 0:
                        aL = aj
                        aR = aj
                    else:
                        if (aR - aL) * (aj - a_avg) > a_delta**2 / 6:
                            aL = 3 * aj - 2 * aR
                        if (aR - aL) * (aj - a_avg) < -(a_delta**2) / 6:
                            aR = 3 * aj - 2 * aL

                    U_L[j, i - 2] = aL
                    U_R[j, i - 1] = aR

                U_R[j, 0] = U_L[j, 0]  # Ensure first interface is consistent
                U_L[j, -1] = U_R[j, -2]  # Ensure last interface is consistent

            return U_L, U_R
        else:
            # Placeholder for 2D/3D PPM reconstruction
            raise NotImplementedError("2D/3D PPM reconstruction not yet implemented")
