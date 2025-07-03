"""
This module defines the Reconstruction class, which handles the spatial
reconstruction of cell-averaged data to the cell interfaces, a key step in
high-resolution finite volume methods.
"""

import numpy as np
from .equation.equation_base import EqnBase
from .limiter import Limiter
from .flux import Flux


class Reconstruction:
    """
    Manages spatial reconstruction methods for 1D finite volume schemes.

    This class provides different strategies to reconstruct the solution values
    from cell centers to their interfaces. This is a crucial step for achieving
    higher-order accuracy. It supports simple first-order (piecewise constant)
    and second-order MUSCL (Monotone Upstream-centered Schemes for Conservation
    Laws) reconstructions.

    The reconstruction can be performed on either primitive or conservative
    variables, as specified by the user.

    Attributes:
        equation (EqnBase): The equation system object, used for state conversions.
        name (str): The name of the reconstruction method (e.g., 'constant', 'muscl').
        in_primitive_domain (bool): If True, reconstruction is performed on
                                    primitive variables; otherwise, on conservative.
        limiter_obj (Limiter | None): An instance of the Limiter class, used for
                                      slope limiting in MUSCL reconstruction.
        flux_obj (Flux): An instance of the Flux class to compute numerical fluxes.
        n_vars (int): The number of variables in the equation system.
        reconst_dicts (dict): Maps reconstruction names to their methods.
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
        """
        Initializes the Reconstruction handler.

        Args:
            eqn_obj (EqnBase): The equation system object.
            str_reconst (str, optional): The reconstruction method to use.
                Supported: 'constant', 'muscl'. Defaults to "constant".
            str_domain (str, optional): The domain for reconstruction.
                Supported: 'primitive', 'conservative'. Defaults to "primitive".
            str_flux (str, optional): The numerical flux scheme to use.
                Defaults to "hllc".
            str_limiter (str, optional): The slope limiter for MUSCL.
                Defaults to "", which implies no limiting (central difference).
            limiter_beta (float, optional): Sharpness parameter for certain limiters.
                Defaults to 1.5.
        """
        if not isinstance(eqn_obj, EqnBase):
            raise TypeError("eqn_obj must be an instance of EqnBase.")

        self.equation = eqn_obj
        self.name = str_reconst.lower()
        self.in_primitive_domain = str_domain.lower() == "primitive"
        self.limiter_obj = (
            Limiter(str_limiter, beta=limiter_beta) if str_limiter else None
        )
        self.flux_obj = Flux(self.equation, str_flux)
        self.n_vars = self.equation.num_vars

        self.reconst_dicts = {
            "constant": self.piecewise_constant,
            "muscl": self.muscl,
            # "ppm": self.ppm,  # Future extension
            # "weno": self.weno, # Future extension
        }

        if self.name not in self.reconst_dicts:
            valid_reconst = list(self.reconst_dicts.keys())
            raise ValueError(
                f"Unsupported reconstruction type: '{self.name}'. "
                f"Choose from {valid_reconst}"
            )

    def reconst_func(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Calls the selected reconstruction method to compute the flux residual.

        Args:
            U (np.ndarray): Array of conservative variables, including ghost cells.
                            Shape: (num_vars, n_cells + 2 * n_ghost).
            dx (np.ndarray): Array of grid cell widths.

        Returns:
            np.ndarray: The residual of the numerical fluxes for each cell.
        """
        return self.reconst_dicts[self.name](U, dx)

    def compute_slopes(self, Q: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Computes the limited slopes of the variable Q for MUSCL reconstruction.

        Args:
            Q (np.ndarray): The variable array (primitive or conservative) to compute
                            slopes for. Shape: (num_vars, n_cells_total).
            dx (np.ndarray): Array of grid cell widths.

        Returns:
            np.ndarray: The limited slope for each variable in each cell.
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

    # --------------------------------------------------- #
    # Reconstruction and Flux Calculation Methods       #
    # --------------------------------------------------- #

    def piecewise_constant(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        First-order piecewise constant reconstruction (Godunov's method).

        The value at the interface is simply the cell-average value from the
        upwind side, leading to a first-order accurate scheme.

        Args:
            U (np.ndarray): Conservative variables array with ghost cells.
            dx (np.ndarray): Grid cell widths.

        Returns:
            np.ndarray: The flux residual for each cell.
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

    def muscl(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Second-order MUSCL reconstruction with slope limiting.

        This method reconstructs the solution to be piecewise linear within each
        cell, achieving second-order accuracy in space.

        Args:
            U (np.ndarray): Conservative variables array with ghost cells.
            dx (np.ndarray): Grid cell widths.

        Returns:
            np.ndarray: The flux residual for each cell.
        """
        N = U.shape[1]
        Flux = np.zeros((self.n_vars, N - 1))
        res = np.zeros((self.n_vars, N))

        if self.in_primitive_domain:
            W = self.equation.to_primitive_batch(U)
            slopes = self.compute_slopes(W, dx)

            # iterate through 1 to N-3, then separately deal with left/right edges
            #   c - cell id, i - interface id
            for i in range(1, N - 2):
                c = i
                W_L = W[:, c] + slopes[:, c] * dx[c] / 2.0
                W_R = W[:, c + 1] - slopes[:, c + 1] * dx[c + 1] / 2.0
                U_L = self.equation.to_conservative(W_L)
                U_R = self.equation.to_conservative(W_R)
                Flux[:, i] = self.flux_obj.flux_func(U_L, U_R)

                # Compute fluxes: assumes U_L and U_R are defined as the left and right states at each interface
                res[:, c] = res[:, c] + Flux[:, i] / dx[c]
                res[:, c + 1] = res[:, c + 1] - Flux[:, i] / dx[c + 1]

            # deal with leftmost face 0
            W_R = W[:, 1] - slopes[:, 1] * dx[1] / 2.0
            W_L = W_R
            U_L = self.equation.to_conservative(W_L)
            U_R = self.equation.to_conservative(W_R)
            Flux[:, 0] = self.flux_obj.flux_func(U_L, U_R)
            res[:, 1] = res[:, 1] - Flux[:, 0] / dx[0]

            # deal with rightmost face N-2
            W_L = W[:, N - 2] + slopes[:, N - 2] * dx[N - 2] / 2.0
            W_R = W_L
            U_L = self.equation.to_conservative(W_L)
            U_R = self.equation.to_conservative(W_R)
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

    def ppm(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Placeholder for the Piecewise Parabolic Method (PPM) reconstruction.

        Args:
            U (np.ndarray): Conservative variables array.
            dx (np.ndarray): Grid cell widths.

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
                imod_delta = True
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

    def weno(self, U: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Placeholder for the Weighted Essentially Non-Oscillatory (WENO) reconstruction.

        Args:
            U (np.ndarray): Conservative variables array.
            dx (np.ndarray): Grid cell widths.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError("WENO reconstruction is not yet implemented.")
