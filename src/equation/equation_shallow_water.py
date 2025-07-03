"""
This module defines the EqnShallowWater class for the 1D Shallow Water
equations, used to model flows in channels, rivers, and coastal regions.
"""

import numpy as np
from .equation_base import EqnBase


class EqnShallowWater(EqnBase):
    """
    Represents the 1D Shallow Water Equations (SWE).

    This system models the conservation of mass (water height) and momentum in a
    one-dimensional channel, assuming a hydrostatic pressure distribution.

    Conservative variables U = [h, hu]
    Primitive variables W = [h, u]

    where:
    - h is the water height
    - u is the water velocity
    - hu is the momentum

    Attributes:
        gravity (float): The acceleration due to gravity.
        var_names (list[str]): Names of the primitive variables.
        num_vars (int): The number of variables in the system (2).
        vel_idx (int): The index of the velocity variable in the primitive state (1).
    """

    def __init__(self, gravity: float = 9.81):
        """
        Initializes the shallow water equation system.

        Args:
            gravity (float, optional): The gravitational acceleration (must be positive).
                                       Defaults to 9.81.
        """
        if gravity <= 0:
            raise ValueError("Gravity must be a positive value.")
        super().__init__(min_value=1e-10)
        self.gravity = gravity
        self.var_names = ["height", "velocity"]
        self.num_vars = len(self.var_names)
        self.vel_idx = 1

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Converts primitive variables [h, u] to conservative variables [h, hu].

        Args:
            W (np.ndarray): A 1D array of primitive variables [h, u].

        Returns:
            np.ndarray: A 1D array of conservative variables [h, hu].
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        h, u = W
        h = np.maximum(h, self.min_value)
        return np.array([h, h * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Converts conservative variables [h, hu] to primitive variables [h, u].

        Args:
            U (np.ndarray): A 1D array of conservative variables [h, hu].

        Returns:
            np.ndarray: A 1D array of primitive variables [h, u].
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        h, hu = U
        h = np.maximum(h, self.min_value)
        u = hu / h
        return np.array([h, u])

    def max_eigenvalue(self, U: np.ndarray) -> float:
        """
        Computes the maximum absolute eigenvalue (|u| + c) of the flux Jacobian.
        Here, c = sqrt(g*h) is the gravity wave speed.

        Args:
            U (np.ndarray): The conservative state vector [h, hu].

        Returns:
            float: The maximum wave speed |u| + c.
        """
        h, hu = U
        h = np.maximum(h, self.min_value)
        sound_speed = np.sqrt(self.gravity * h)
        u = hu / h
        eigenvalue_max = max(u - sound_speed, u + sound_speed)
        return eigenvalue_max

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the physical flux vector F(U) for the shallow water equations.

        F(U) = [hu, hu² + 0.5*g*h²]

        Args:
            U (np.ndarray): The conservative state vector [h, hu].

        Returns:
            np.ndarray: The physical flux vector.
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        h, u = self.to_primitive(U)
        flux_mass = h * u
        flux_momentum = h * u**2 + 0.5 * self.gravity * h**2
        return np.array([flux_mass, flux_momentum])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple[float, float]:
        """
        Computes Roe-averaged quantities for the shallow water equations.

        Args:
            U_L (np.ndarray): Left conservative state [h, hu]_L.
            U_R (np.ndarray): Right conservative state [h, hu]_R.

        Returns:
            tuple[float, float]: A tuple containing the Roe-averaged velocity
                                 and wave speed (u_roe, c_roe).
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        hL, huL = self.to_primitive(U_L)
        hL = np.maximum(hL, self.min_value)
        uL = huL / hL

        # right state
        hR, huR = self.to_primitive(U_R)
        hR = np.maximum(hR, self.min_value)
        uR = huR / hR

        # Roe averages
        sqrt_hL = np.sqrt(hL)
        sqrt_hR = np.sqrt(hR)

        u_roe = (uL * sqrt_hL + uR * sqrt_hR) / np.maximum(
            sqrt_hL + sqrt_hR, self.min_value
        )

        h_roe = sqrt_hL * sqrt_hR
        h_roe = np.maximum(h_roe, self.min_value)

        c_roe = np.sqrt(self.gravity * h_roe)

        return u_roe, c_roe

    # ---------------------------------------------------- #
    # Numerical Flux Methods for Shallow Water Equations   #
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Roe numerical flux for the shallow water equations.

        Args:
            U_L (np.ndarray): Left conservative state [h, hu]_L.
            U_R (np.ndarray): Right conservative state [h, hu]_R.

        Returns:
            np.ndarray: The Roe numerical flux vector.
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        hL, uL = self.to_primitive(U_L)
        aL = np.sqrt(self.gravity * hL)

        # right state
        hR, uR = self.to_primitive(U_R)
        aR = np.sqrt(self.gravity * hR)

        # Roe averages
        sqrt_hL = np.sqrt(hL)
        sqrt_hR = np.sqrt(hR)
        u_roe = (uL * sqrt_hL + uR * sqrt_hR) / np.maximum(
            sqrt_hL + sqrt_hR, self.min_value
        )
        h_roe = sqrt_hL * sqrt_hR
        h_roe = np.maximum(h_roe, self.min_value)
        c_roe = np.sqrt(self.gravity * h_roe)

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # Differences in primitive variables
        dh = hR - hL
        dhu = U_R[1] - U_L[1]

        # Wave strengths (characteristic variables)
        alphaMat = np.array(
            [
                (dhu - u_roe * dh - c_roe * dh) / (-2 * c_roe),
                (dhu - u_roe * dh + c_roe * dh) / (2 * c_roe),
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
        F = (FL + FR - R @ (lambdas * alphaMat)) / 2.0

        return F

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Placeholder for AUSM flux. Currently not implemented for SWE.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: An array of zeros.
        """
        # NOTE: This is a placeholder. A proper AUSM implementation for SWE
        # would be more complex.
        return np.zeros(self.num_vars)

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLC numerical flux for the shallow water equations.

        Args:
            U_L (np.ndarray): Left conservative state [h, hu]_L.
            U_R (np.ndarray): Right conservative state [h, hu]_R.

        Returns:
            np.ndarray: The HLLC numerical flux vector.
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)
        hL, uL = W_L
        hR, uR = W_R
        hL = np.maximum(hL, self.min_value)
        hR = np.maximum(hR, self.min_value)

        cL = np.sqrt(self.gravity * hL)
        cR = np.sqrt(self.gravity * hR)
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)
        denom = hR * (SR - uR) - hL * (SL - uL)
        S_star = (
            hR * uR * (SR - uR)
            - hL * uL * (SL - uL)
            + 0.5 * self.gravity * (hR**2 - hL**2)
        ) / (denom + self.min_value)

        hL_star = np.maximum(
            hL * (SL - uL) / (SL - S_star + self.min_value), self.min_value
        )
        hR_star = np.maximum(
            hR * (SR - uR) / (SR - S_star + self.min_value), self.min_value
        )
        UL_star = np.array([hL_star, hL_star * S_star])
        UR_star = np.array([hR_star, hR_star * S_star])

        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if SL >= 0:
            F = FL
        elif SL <= 0 <= S_star:
            F = FL + SL * (UL_star - U_L)
        elif S_star <= 0 <= SR:
            F = FR + SR * (UR_star - U_R)
        else:
            F = FR

        return F

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute the HLLC numerical flux for the shallow water equations.

        Args:
            U_L (np.ndarray): Left conservative state [h, hu]_L.
            U_R (np.ndarray): Right conservative state [h, hu]_R.

        Returns:
            np.ndarray: The HLLE numerical flux vector.
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)
        hL, uL = W_L
        hR, uR = W_R
        hL = np.maximum(hL, self.min_value)
        hR = np.maximum(hR, self.min_value)

        cL = np.sqrt(self.gravity * hL)
        cR = np.sqrt(self.gravity * hR)
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)
        denom = hR * (SR - uR) - hL * (SL - uL)
        S_star = (
            hR * uR * (SR - uR)
            - hL * uL * (SL - uL)
            + 0.5 * self.gravity * (hR**2 - hL**2)
        ) / (denom + self.min_value)

        hL_star = np.maximum(
            hL * (SL - uL) / (SL - S_star + self.min_value), self.min_value
        )
        hR_star = np.maximum(
            hR * (SR - uR) / (SR - S_star + self.min_value), self.min_value
        )
        UL_star = np.array([hL_star, hL_star * S_star])
        UR_star = np.array([hR_star, hR_star * S_star])

        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if SL >= 0:
            F = FL
        elif SL <= 0 <= S_star:
            F = FL + SL * (UL_star - U_L)
        elif S_star <= 0 <= SR:
            F = FR + SR * (UR_star - U_R)
        else:
            F = FR

        return F
