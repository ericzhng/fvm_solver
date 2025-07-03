"""
This module defines the EqnKK class for the Keyfitz-Kranzer system, a
non-hyperbolic system of conservation laws.
"""

import numpy as np
from .equation_base import EqnBase


class EqnKK(EqnBase):
    """
    Represents the 1D Keyfitz-Kranzer (KK) system of equations.

    This is a non-strictly hyperbolic system given by:
    ∂u/∂t + ∂(r*u)/∂x = 0
    ∂v/∂t + ∂(r*v)/∂x = 0
    where r = sqrt(u² + v²).

    The variables u and v are the conserved quantities, and for this system,
    the primitive and conservative variables are the same.

    Attributes:
        var_names (list[str]): The names of the variables, ["u", "v"].
        num_vars (int): The number of variables in the system (2).
        vel_idx (None): Index of the velocity variable. Not applicable in this context.
    """

    def __init__(self):
        """
        Initializes the Keyfitz-Kranzer (KK) equation system.
        """
        super().__init__(min_value=1e-10)
        self.var_names = ["u", "v"]
        self.num_vars = len(self.var_names)
        self.vel_idx = None  # No single velocity component

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Converts primitive variables to conservative variables.
        For the KK system, they are identical.

        Args:
            W (np.ndarray): Array of primitive variables [u, v].

        Returns:
            np.ndarray: Array of conservative variables [u, v].
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        return W.copy()

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Converts conservative variables to primitive variables.
        For the KK system, they are identical.

        Args:
            U (np.ndarray): Array of conservative variables [u, v].

        Returns:
            np.ndarray: Array of primitive variables [u, v].
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        return U.copy()

    def max_eigenvalue(self, U: np.ndarray) -> float:
        """
        Computes the maximum absolute eigenvalue of the flux Jacobian.
        For the KK system, the eigenvalues are 0 and 2r.

        Args:
            U (np.ndarray): The state vector [u, v].

        Returns:
            float: The maximum eigenvalue, 2r.
        """
        r = np.sqrt(U[0] ** 2 + U[1] ** 2)
        return 2.0 * r

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the physical flux F(U) for the Keyfitz-Kranzer system.
        F(U) = [r*u, r*v], where r = sqrt(u² + v²).

        Args:
            U (np.ndarray): The state vector [u, v].

        Returns:
            np.ndarray: The physical flux vector.
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        r = np.sqrt(U[0] ** 2 + U[1] ** 2)
        return np.array([U[0] * r, U[1] * r])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple[float, int]:
        """
        Computes Roe-averaged quantities for the Keyfitz-Kranzer system.

        Args:
            U_L (np.ndarray): Left state [u, v]_L.
            U_R (np.ndarray): Right state [u, v]_R.

        Returns:
            tuple[float, int]: A tuple containing the Roe-averaged radius (r_roe)
                               and a placeholder value (0).
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # Left and right states
        u_L, v_L = U_L
        u_R, v_R = U_R
        r_L = np.sqrt(u_L**2 + v_L**2)
        r_R = np.sqrt(u_R**2 + v_R**2)

        # Roe averages (using a simple arithmetic mean for r)
        sqrt_rL = np.sqrt(r_L)
        sqrt_rR = np.sqrt(r_R)
        u_roe = (u_L * sqrt_rL + u_R * sqrt_rR) / np.maximum(
            sqrt_rL + sqrt_rR, self.min_value
        )
        v_roe = (v_L * sqrt_rL + v_R * sqrt_rR) / np.maximum(
            sqrt_rL + sqrt_rR, self.min_value
        )
        r_roe = np.sqrt(u_roe * u_roe + v_roe * v_roe)

        # r_roe = (r_L + r_R) / 2.0

        return r_roe, 0

    # ---------------------------------------------------- #
    # Flux methods for the Keyfitz-Kranzer system          #
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Roe numerical flux for the Keyfitz-Kranzer system.

        Args:
            U_L (np.ndarray): Left state [u, v]_L.
            U_R (np.ndarray): Right state [u, v]_R.

        Returns:
            np.ndarray: The Roe numerical flux vector.
        """
        # Physical fluxes at left and right states
        F_L = self.compute_flux(U_L)
        F_R = self.compute_flux(U_R)

        # left state
        uL, vL = U_L
        uR, vR = U_R
        rL = np.sqrt(U_L[0] * U_L[0] + U_L[1] * U_L[1])
        rR = np.sqrt(U_R[0] * U_R[0] + U_R[1] * U_R[1])

        # Roe averages
        sqrt_rL = np.sqrt(rL)
        sqrt_rR = np.sqrt(rR)
        u_roe = (uL * sqrt_rL + uR * sqrt_rR) / np.maximum(
            sqrt_rL + sqrt_rR, self.min_value
        )
        v_roe = (vL * sqrt_rL + vR * sqrt_rR) / np.maximum(
            sqrt_rL + sqrt_rR, self.min_value
        )
        vec = np.array([u_roe, v_roe])
        r_roe = np.sqrt(u_roe * u_roe + v_roe * v_roe)

        # Differences in primitive variables
        du = uR - uL
        dv = vR - vL

        # eigenvalues and eigenvectors
        lambda2 = 2 * r_roe

        # wave strength
        alpha = (u_roe * du + v_roe * dv) / r_roe**2

        # Add the matrix dissipation term to complete the Roe flux
        roe_flux = (F_L + F_R - (lambda2 * alpha * vec)) / 2

        return roe_flux

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes an upwind-style flux based on the average state.
        This is a placeholder and not a standard AUSM implementation.

        Args:
            U_L (np.ndarray): Left state [u, v]_L.
            U_R (np.ndarray): Right state [u, v]_R.

        Returns:
            np.ndarray: The numerical flux vector.
        """
        # A simple upwind scheme based on the average state.
        # This is not a standard AUSM flux for this system.
        U_avg = (U_L + U_R) / 2.0
        # The "velocity" for upwinding is taken as the magnitude r.
        if U_avg[0] >= 0:  # Arbitrary condition, not physically based
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Placeholder for HLLC flux. Returns a simple upwind flux.

        Args:
            U_L (np.ndarray): Left state [u, v]_L.
            U_R (np.ndarray): Right state [u, v]_R.

        Returns:
            np.ndarray: The numerical flux vector.
        """
        # This is not a standard HLLC flux for this system.
        U_avg = (U_L + U_R) / 2.0
        if U_avg[0] >= 0:  # Arbitrary condition, not physically based
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLE flux for the Keyfitz-Kranzer system.

        Args:
            U_L (np.ndarray): Left state [u, v]_L.
            U_R (np.ndarray): Right state [u, v]_R.

        Returns:
            np.ndarray: The HLLE numerical flux vector.
        """
        F_L = self.compute_flux(U_L)
        F_R = self.compute_flux(U_R)

        r_L = np.sqrt(U_L[0] ** 2 + U_L[1] ** 2)
        r_R = np.sqrt(U_R[0] ** 2 + U_R[1] ** 2)

        # Estimate wave speeds
        S_L = -2 * max(r_L, r_R)
        S_R = 2 * max(r_L, r_R)

        # HLLE flux formula
        hlle_flux = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)

        return hlle_flux
