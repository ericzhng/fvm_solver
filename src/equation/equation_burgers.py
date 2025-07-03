"""
This module defines the EqnBurgers class for the 1D inviscid Burgers' equation.
"""

import numpy as np
from .equation_base import EqnBase


class EqnBurgers(EqnBase):
    """
    Represents the 1D Inviscid Burgers' Equation: ∂u/∂t + ∂(u²/2)/∂x = 0.

    This is a simple non-linear scalar conservation law that can develop shocks.
    For Burgers' equation, the primitive and conservative variables are the same,
    representing the velocity 'u'.

    Attributes:
        var_names (list[str]): The name of the variable, which is ["velocity"].
        num_vars (int): The number of variables in the system (1).
        vel_idx (int): The index of the velocity variable in the state vector (0).
    """

    def __init__(self):
        """
        Initializes the EqnBurgers object.
        """
        super().__init__(min_value=1e-10)
        self.var_names = ["velocity"]
        self.num_vars = len(self.var_names)
        self.vel_idx = 0

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Converts primitive variables to conservative variables.
        For Burgers' equation, they are identical.

        Args:
            W (np.ndarray): Array of primitive variables [u]. Shape: (1,).

        Returns:
            np.ndarray: Array of conservative variables [u]. Shape: (1,).
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        return W.copy()

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Converts conservative variables to primitive variables.
        For Burgers' equation, they are identical.

        Args:
            U (np.ndarray): Array of conservative variables [u]. Shape: (1,).

        Returns:
            np.ndarray: Array of primitive variables [u]. Shape: (1,).
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        return U.copy()

    def max_eigenvalue(self, U: np.ndarray) -> float:
        """
        Computes the maximum eigenvalue (characteristic speed) of the system.
        For Burgers' equation, the characteristic speed is simply the velocity 'u'.

        Args:
            U (np.ndarray): The conservative state vector [u].

        Returns:
            float: The characteristic speed, u.
        """
        # The eigenvalue is the velocity u itself.
        return U[0]

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the physical flux F(U) for Burgers' equation.
        The flux is given by F(u) = u² / 2.

        Args:
            U (np.ndarray): The conservative state vector [u]. Shape: (1,).

        Returns:
            np.ndarray: The flux vector [u²/2]. Shape: (1,).
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        # Flux F(u) = u^2 / 2
        return np.array([U[0] ** 2 / 2.0])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple[float, float]:
        """
        Computes the Roe-averaged velocity for Burgers' equation.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            tuple[float, float]: A tuple containing the Roe-averaged velocity (u_roe)
                               and a placeholder value (0).
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # Left & right states
        uL = U_L[0]
        uR = U_R[0]
        # Roe-averaged velocity is the arithmetic mean.
        u_roe = (uL + uR) / 2.0

        return u_roe, 0

    # ---------------------------------------------------- #
    # Flux methods specific to Burgers' equation           #
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Roe numerical flux for Burgers' equation.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            np.ndarray: The Roe numerical flux.
        """
        # Left and Right physical fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # Left & right states
        uL = U_L[0]
        uR = U_R[0]
        # Roe-averaged velocity (eigenvalue)
        u_roe = (uL + uR) / 2.0

        # Difference in the state variable
        du = uR - uL

        # eigenvalues and eigenvectors
        lambda1 = u_roe

        # wave strength
        alpha = du

        # Add the matrix dissipation term to complete the Roe flux
        roe_flux = (FL + FR - (lambda1 * alpha)) / 2

        return roe_flux

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the AUSM flux. For Burgers' equation, this is implemented
        as a simple upwind flux based on the Roe-averaged velocity.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            np.ndarray: The numerical flux.
        """
        # left & right states
        uL = U_L[0]
        uR = U_R[0]
        # Use Roe-averaged velocity to determine the upwind direction.
        u_roe = (uL + uR) / 2.0

        if u_roe >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLC flux. For the scalar Burgers' equation, this
        simplifies to the Godunov flux, which is equivalent to an upwind flux.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            np.ndarray: The numerical flux.
        """
        # For scalar equations, HLLC is the same as Godunov/upwind flux.
        uL = U_L[0]
        uR = U_R[0]
        u_roe = (uL + uR) / 2.0

        if u_roe >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLE flux. For the scalar Burgers' equation, this
        simplifies to the Godunov flux, which is equivalent to an upwind flux.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            np.ndarray: The numerical flux.
        """
        # For scalar equations, HLLE is the same as Godunov/upwind flux.
        uL = U_L[0]
        uR = U_R[0]
        u_roe = (uL + uR) / 2.0

        if u_roe >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)
