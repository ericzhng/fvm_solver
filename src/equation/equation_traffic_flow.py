"""
This module defines the EqnTrafficLWR class for the Lighthill-Whitham-Richards
(LWR) model of traffic flow.
"""

import numpy as np
from .equation_base import EqnBase


class EqnTrafficLWR(EqnBase):
    """
    Represents the 1D Lighthill-Whitham-Richards (LWR) traffic flow model.

    This is a scalar, non-linear conservation law that models the density of
    cars on a road. The equation is given by:
    ∂ρ/∂t + ∂F(ρ)/∂x = 0
    where F(ρ) = ρ * v(ρ) is the flux (flow rate of cars).

    The car velocity v(ρ) is assumed to be a linearly decreasing function of
    the density ρ: v(ρ) = v_max * (1 - ρ / ρ_max).

    The conservative and primitive variable is the car density, ρ.

    Attributes:
        rho_max (float): The maximum possible car density (jam density).
        v_max (float): The maximum speed of a car in free-flow traffic.
        var_names (list[str]): The name of the variable, ["density"].
        num_vars (int): The number of variables in the system (1).
        vel_idx (None): Not applicable for this scalar equation.
    """

    def __init__(self, rho_max: float = 1.0, v_max: float = 1.0):
        """
        Initializes the LWR traffic flow model.

        Args:
            rho_max (float, optional): Maximum car density. Defaults to 1.0.
            v_max (float, optional): Maximum car speed. Defaults to 1.0.
        """
        super().__init__(min_value=1e-10)
        self.rho_max = rho_max
        self.v_max = v_max
        self.var_names = ["density"]
        self.num_vars = len(self.var_names)
        self.vel_idx = None

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Converts primitive variables to conservative variables.
        For the LWR model, they are identical (ρ).

        Args:
            W (np.ndarray): Array of primitive variables [ρ].

        Returns:
            np.ndarray: Array of conservative variables [ρ].
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        return W.copy()

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Converts conservative variables to primitive variables.
        For the LWR model, they are identical (ρ).

        Args:
            U (np.ndarray): Array of conservative variables [ρ].

        Returns:
            np.ndarray: Array of primitive variables [ρ].
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        return U.copy()

    def max_eigenvalue(self, U: np.ndarray) -> float:
        """
        Computes the maximum eigenvalue (characteristic speed) of the system.
        This is the derivative of the flux with respect to density, dF/dρ.

        Args:
            U (np.ndarray): The conservative state vector [ρ].

        Returns:
            float: The characteristic speed.
        """
        rho = U[0]
        # Eigenvalue = dF/dρ = v_max * (1 - 2*ρ / ρ_max)
        return self.v_max * (1 - 2.0 * rho / self.rho_max)

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the physical flux F(ρ) for the LWR model.
        F(ρ) = ρ * v_max * (1 - ρ / ρ_max).

        Args:
            U (np.ndarray): The conservative state vector [ρ].

        Returns:
            np.ndarray: The physical flux vector.
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho = U[0]
        flux = rho * self.v_max * (1 - rho / self.rho_max)
        return np.array([flux])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple[float, float]:
        """
        Computes the Roe-averaged density for the LWR model.

        Args:
            U_L (np.ndarray): Left state [ρ_L].
            U_R (np.ndarray): Right state [ρ_R].

        Returns:
            tuple[float, float]: A tuple containing the Roe-averaged density (ρ_roe)
                               and a placeholder value (0).
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        rho_L = U_L[0]
        rho_R = U_R[0]
        rho_roe = (rho_L + rho_R) / 2.0

        return rho_roe, 0

    # ---------------------------------------------------- #
    # Flux methods for the LWR Traffic Flow Model          #
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Roe numerical flux for the LWR model.

        Args:
            U_L (np.ndarray): Left state [ρ_L].
            U_R (np.ndarray): Right state [ρ_R].

        Returns:
            np.ndarray: The Roe numerical flux.
        """
        F_L = self.compute_flux(U_L)
        F_R = self.compute_flux(U_R)

        rho_L = U_L[0]
        rho_R = U_R[0]
        rho_roe = (rho_L + rho_R) / 2.0

        # Roe-averaged eigenvalue
        lambda_roe = self.v_max * (1 - 2 * rho_roe / self.rho_max)

        # Difference in the state variable
        d_rho = rho_R - rho_L

        # Roe flux formula
        roe_flux = 0.5 * (F_L + F_R) - 0.5 * abs(lambda_roe) * d_rho

        return roe_flux

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes an upwind flux based on the Roe-averaged characteristic speed.

        Args:
            U_L (np.ndarray): Left state [ρ_L].
            U_R (np.ndarray): Right state [ρ_R].

        Returns:
            np.ndarray: The numerical flux.
        """
        rho_L = U_L[0]
        rho_R = U_R[0]
        rho_roe = (rho_L + rho_R) / 2.0
        lambda_roe = self.v_max * (1 - 2 * rho_roe / self.rho_max)

        if lambda_roe >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        For this scalar equation, HLLC simplifies to the Godunov flux,
        which is implemented here as an upwind flux.

        Args:
            U_L (np.ndarray): Left state [ρ_L].
            U_R (np.ndarray): Right state [ρ_R].

        Returns:
            np.ndarray: The numerical flux.
        """
        rho_L = U_L[0]
        rho_R = U_R[0]
        rho_roe = (rho_L + rho_R) / 2.0
        lambda_roe = self.v_max * (1 - 2 * rho_roe / self.rho_max)

        if lambda_roe >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLE flux for the LWR model.

        Args:
            U_L (np.ndarray): Left state [ρ_L].
            U_R (np.ndarray): Right state [ρ_R].

        Returns:
            np.ndarray: The HLLE numerical flux.
        """
        F_L = self.compute_flux(U_L)
        F_R = self.compute_flux(U_R)

        lambda_L = self.max_eigenvalue(U_L)
        lambda_R = self.max_eigenvalue(U_R)

        # Estimate wave speeds
        S_L = min(lambda_L, lambda_R)
        S_R = max(lambda_L, lambda_R)

        # HLLE flux formula
        if S_L >= 0:
            return F_L
        elif S_R <= 0:
            return F_R
        else:
            return (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
