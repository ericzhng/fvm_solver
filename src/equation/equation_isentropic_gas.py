"""
This module defines the EqnIsentropicGas class for the 1D isentropic
Euler equations, a simplified model for gas dynamics.
"""

import numpy as np
from .equation_base import EqnBase


class EqnIsentropicGas(EqnBase):
    """
    Represents the 1D isentropic gas dynamics equations.

    This system models the conservation of mass and momentum, assuming that the
    flow process is reversible and adiabatic (no heat exchange), which implies
    constant entropy. The pressure is related to density via the isentropic
    relation p = k * ρ^γ.

    Conservative variables U = [ρ, ρu]
    Primitive variables W = [ρ, u]

    where:
    - ρ is the density
    - u is the velocity

    Attributes:
        gamma (float): The ratio of specific heats (adiabatic index).
        k (float): The polytropic constant in the pressure-density relation.
        var_names (list[str]): Names of the primitive variables.
        num_vars (int): The number of variables in the system (2).
        vel_idx (int): The index of the velocity variable in the primitive state (1).
    """

    def __init__(self, gamma: float = 1.4, k: float = 1.0):
        """
        Initializes the isentropic gas equation system.

        Args:
            gamma (float, optional): The ratio of specific heats. Defaults to 1.4.
            k (float, optional): The polytropic constant. Defaults to 1.0.
        """
        super().__init__(min_value=1e-10)
        self.gamma = gamma
        self.k = k
        self.var_names = ["density", "velocity"]
        self.num_vars = len(self.var_names)
        self.vel_idx = 1

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Converts primitive variables [ρ, u] to conservative variables [ρ, ρu].

        Args:
            W (np.ndarray): A 1D array of primitive variables [ρ, u].

        Returns:
            np.ndarray: A 1D array of conservative variables [ρ, ρu].
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        rho, u = W
        rho = np.maximum(rho, self.min_value)
        return np.array([rho, rho * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Converts conservative variables [ρ, ρu] to primitive variables [ρ, u].

        Args:
            U (np.ndarray): A 1D array of conservative variables [ρ, ρu].

        Returns:
            np.ndarray: A 1D array of primitive variables [ρ, u].
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, m = U
        rho = np.maximum(rho, self.min_value)
        u = m / rho
        return np.array([rho, u])

    def max_eigenvalue(self, U: np.ndarray) -> float:
        """
        Computes the maximum absolute eigenvalue (|u| + c) of the flux Jacobian.

        Args:
            U (np.ndarray): The conservative state vector [ρ, ρu].

        Returns:
            float: The maximum wave speed |u| + c.
        """
        rho, u = self.to_primitive(U)
        sound_speed = np.sqrt(self.gamma * self.k * rho ** (self.gamma - 1))
        eigenvalue_max = max(u - sound_speed, u + sound_speed)
        return eigenvalue_max

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the physical flux vector F(U) for the isentropic equations.

        F(U) = [ρu, ρu² + p], where p = k * ρ^γ.

        Args:
            U (np.ndarray): The conservative state vector [ρ, ρu].

        Returns:
            np.ndarray: The physical flux vector.
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, u = self.to_primitive(U)
        p = self.k * rho**self.gamma
        return np.array([rho * u, rho * u**2 + p])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple[float, float]:
        """
        Computes Roe-averaged quantities for the isentropic gas equations.

        Args:
            U_L (np.ndarray): Left conservative state [ρ, ρu]_L.
            U_R (np.ndarray): Right conservative state [ρ, ρu]_R.

        Returns:
            tuple[float, float]: A tuple containing the Roe-averaged velocity
                                 and sound speed (u_roe, c_roe).
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        rho_L, u_L = self.to_primitive(U_L)
        rho_R, u_R = self.to_primitive(U_R)

        sqrt_rho_L = np.sqrt(rho_L)
        sqrt_rho_R = np.sqrt(rho_R)
        denom = sqrt_rho_L + sqrt_rho_R

        u_roe = (u_L * sqrt_rho_L + u_R * sqrt_rho_R) / np.maximum(
            denom, self.min_value
        )
        rho_roe = np.sqrt(rho_L * rho_R)
        c_roe = np.sqrt(self.gamma * self.k * rho_roe ** (self.gamma - 1))

        return u_roe, c_roe

    # ---------------------------------------------------- #
    # Numerical Flux Methods for Isentropic Gas Dynamics   #
    # ---------------------------------------------------- #

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLE numerical flux for the isentropic gas equations.

        Args:
            U_L (np.ndarray): Left conservative state [ρ, ρu]_L.
            U_R (np.ndarray): Right conservative state [ρ, ρu]_R.

        Returns:
            np.ndarray: The HLLE numerical flux vector.
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)

        # Extract variables
        rhoL, uL = W_L
        rhoR, uR = W_R
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)

        # Compute wave speeds
        cL = np.sqrt(self.gamma * self.k * W_L[0] ** (self.gamma - 1))
        cR = np.sqrt(self.gamma * self.k * W_R[0] ** (self.gamma - 1))
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = (
            0.5 * (uL + uR)
            if abs(denom) < self.min_value
            else (pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / denom
        )

        # Compute intermediate states
        rho_star_L = max(
            rhoL * (S_L - uL) / (S_L - S_star + self.min_value), self.min_value
        )
        rho_star_R = max(
            rhoR * (S_R - uR) / (S_R - S_star + self.min_value), self.min_value
        )
        U_L_star = np.array([rho_star_L, rho_star_L * S_star])
        U_R_star = np.array([rho_star_R, rho_star_R * S_star])

        # Compute flux
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if S_L >= 0:
            F = FL
        elif S_L <= 0 <= S_star:
            F = FL + S_L * (U_L_star - U_L)
        elif S_star <= 0 <= S_R:
            F = FR + S_R * (U_R_star - U_R)
        else:
            F = FR

        return F

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Roe numerical flux for the isentropic gas equations.

        Args:
            U_L (np.ndarray): Left conservative state [ρ, ρu]_L.
            U_R (np.ndarray): Right conservative state [ρ, ρu]_R.

        Returns:
            np.ndarray: The Roe numerical flux vector.
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)
        rhoL, uL = W_L
        rhoR, uR = W_R
        aL = np.sqrt(self.gamma * self.k * rhoL ** (self.gamma - 1))
        aR = np.sqrt(self.gamma * self.k * rhoR ** (self.gamma - 1))

        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (
            np.sqrt(rhoL) + np.sqrt(rhoR) + self.min_value
        )
        rho_roe = np.maximum(rho_roe, self.min_value)
        c_roe = np.sqrt(self.gamma * self.k * rho_roe ** (self.gamma - 1))

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # Differences in primitive variables
        dr = rhoR - rhoL
        dhu = U_R[1] - U_L[1]

        # Wave strengths (characteristic variables)
        alphaMat = np.array(
            [
                (dhu - u_roe * dr - c_roe * dr) / (-2 * c_roe),
                (dhu - u_roe * dr + c_roe * dr) / (2 * c_roe),
            ]
        )

        # Absolute values of the wave speeds (Eigenvalues)
        lambdas = np.array([abs(u_roe - c_roe), abs(u_roe + c_roe)])

        # Harten's Entropy Fix JCP(1983), 49, pp357-393
        Da = max(0, 4 * ((uR - aR) - (uL - aL)))
        if lambdas[0] < Da / 2 and Da != 0:
            lambdas[0] = lambdas[0] ** 2 / Da + Da / 4

        Da = max(0, 4 * ((uR + aR) - (uL + aL)))
        if lambdas[1] < Da / 2 and Da != 0:
            lambdas[1] = lambdas[1] ** 2 / Da + Da / 4

        # Right eigenvectors
        R = np.array(
            [
                [1, 1],
                [u_roe - c_roe, u_roe + c_roe],
            ]
        )

        # Add the matrix dissipation term to complete the Roe flux
        roe_flux = (FL + FR - R @ (lambdas * alphaMat)) / 2

        return roe_flux

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLC numerical flux for the isentropic gas equations.

        Args:
            U_L (np.ndarray): Left conservative state [ρ, ρu]_L.
            U_R (np.ndarray): Right conservative state [ρ, ρu]_R.

        Returns:
            np.ndarray: The HLLC numerical flux vector.
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)

        # Extract variables
        rhoL, uL = W_L
        rhoR, uR = W_R
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)

        # Compute wave speeds
        cL = np.sqrt(self.gamma * self.k * W_L[0] ** (self.gamma - 1))
        cR = np.sqrt(self.gamma * self.k * W_R[0] ** (self.gamma - 1))
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = (
            0.5 * (uL + uR)
            if abs(denom) < self.min_value
            else (pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / denom
        )

        # Compute intermediate states
        rho_star_L = max(
            rhoL * (S_L - uL) / (S_L - S_star + self.min_value), self.min_value
        )
        rho_star_R = max(
            rhoR * (S_R - uR) / (S_R - S_star + self.min_value), self.min_value
        )
        U_L_star = np.array([rho_star_L, rho_star_L * S_star])
        U_R_star = np.array([rho_star_R, rho_star_R * S_star])

        # Compute flux
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if S_L >= 0:
            F = FL
        elif S_L <= 0 <= S_star:
            F = FL + S_L * (U_L_star - U_L)
        elif S_star <= 0 <= S_R:
            F = FR + S_R * (U_R_star - U_R)
        else:
            F = FR

        return F

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLE numerical flux for the isentropic gas equations.

        Args:
            U_L (np.ndarray): Left conservative state [ρ, ρu]_L.
            U_R (np.ndarray): Right conservative state [ρ, ρu]_R.

        Returns:
            np.ndarray: The HLLE numerical flux vector.
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)

        # Extract variables
        rhoL, uL = W_L
        rhoR, uR = W_R
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)

        # Compute wave speeds
        cL = np.sqrt(self.gamma * self.k * W_L[0] ** (self.gamma - 1))
        cR = np.sqrt(self.gamma * self.k * W_R[0] ** (self.gamma - 1))
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = (
            0.5 * (uL + uR)
            if abs(denom) < self.min_value
            else (pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / denom
        )

        # Compute intermediate states
        rho_star_L = max(
            rhoL * (S_L - uL) / (S_L - S_star + self.min_value), self.min_value
        )
        rho_star_R = max(
            rhoR * (S_R - uR) / (S_R - S_star + self.min_value), self.min_value
        )
        U_L_star = np.array([rho_star_L, rho_star_L * S_star])
        U_R_star = np.array([rho_star_R, rho_star_R * S_star])

        # Compute flux
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if S_L >= 0:
            F = FL
        elif S_L <= 0 <= S_star:
            F = FL + S_L * (U_L_star - U_L)
        elif S_star <= 0 <= S_R:
            F = FR + S_R * (U_R_star - U_R)
        else:
            F = FR

        return F
