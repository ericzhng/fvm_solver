"""
This module defines the EquationAdvection class for the 1D linear advection equation.
"""

import numpy as np
from .equation_base import EqnBase


class EqnAdvection(EqnBase):
    """
    Represents the 1D Linear Advection Equation: ∂u/∂t + a ∂u/∂x = 0.

    In this equation, 'u' is the transported quantity and 'a' is the constant
    advection speed. For the linear advection equation, the primitive and
    conservative variables are identical.

    Attributes:
        speed (float): The constant advection speed 'a'.
        var_names (list[str]): A list containing the name of the variable, e.g., ["quantity"].
        num_vars (int): The number of variables in the system (which is 1).
        vel_idx (None): Index of the velocity variable. Not applicable for advection.
    """

    def __init__(self, speed: float = 1.0):
        """
        Initializes the EqnAdvection object.

        Args:
            speed (float, optional): The constant advection speed 'a'. Defaults to 1.0.
        """
        super().__init__(min_value=1e-10)
        self.speed = speed
        self.var_names = ["quantity"]
        self.num_vars = len(self.var_names)
        self.vel_idx = None  # No velocity component in this equation

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Converts primitive variables to conservative variables.

        For the linear advection equation, primitive and conservative
        variables are the same.

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

        For the linear advection equation, conservative and primitive
        variables are the same.

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
        Computes the maximum eigenvalue of the Jacobian matrix.

        For the linear advection equation, the only eigenvalue is the
        advection speed 'a'. Note: This method returns the raw speed, not
        its absolute value.

        Args:
            U (np.ndarray): The conservative state vector [u]. Not used for
                            linear advection but required for compatibility.

        Returns:
            float: The eigenvalue (wave speed), which is 'a'.
        """
        # For linear advection, the "wave speed" is a
        return self.speed

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the flux F(U) for the advection equation.

        The flux is given by F(u) = a * u.

        Args:
            U (np.ndarray): The conservative state vector [u]. Shape: (1,).

        Returns:
            np.ndarray: The flux vector [a*u]. Shape: (1,).
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        # Flux F(U) = a * U
        return np.array([self.speed * U[0]])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple[float, float]:
        """
        Computes Roe-averaged states for the advection equation.

        For linear advection, this is a simple arithmetic mean of the states.
        The second return value is a placeholder, returning 0.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            tuple[float, float]: A tuple containing the Roe-averaged state (u_roe)
                               and a placeholder value (0.0).
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left & right states
        uL = U_L[0]
        uR = U_R[0]
        # For linear advection, the Roe average is the arithmetic mean.
        u_roe = (uL + uR) / 2.0

        # The second value is a placeholder, analogous to sound speed in other systems.
        return u_roe, 0

    # ---------------------------------------------------- #
    # Flux methods specific to the advection equation      #
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Roe numerical flux for the advection equation.

        Note: The current implementation simplifies to the upwind flux
        F(U_L) if the advection speed `a` is positive, and does not
        represent the standard Roe flux formula.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            np.ndarray: The numerical flux.
        """
        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # left & right states
        uL = U_L[0]
        uR = U_R[0]

        # Differences in primitive variables
        du = uR - uL

        # Eigenvalue (wave speed)
        lambda1 = self.speed

        # Wave strength
        alpha = du

        # This formula simplifies to the upwind flux F(U_L)
        Roe = (FL + FR - (lambda1 * alpha)) / 2

        return Roe

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the AUSM flux. For linear advection, this is implemented
        as a simple upwind flux.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            np.ndarray: The numerical flux, calculated using the upwind method.
        """
        # For linear advection, this is the standard upwind flux.
        a = self.speed
        if a >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLC flux. For linear advection, this is implemented
        as a simple upwind flux.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            np.ndarray: The numerical flux, calculated using the upwind method.
        """
        # For linear advection, HLLC simplifies to the upwind flux.
        a = self.speed
        if a >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLE flux. For linear advection, this is implemented
        as a simple upwind flux.

        Args:
            U_L (np.ndarray): Left conservative state [u_L].
            U_R (np.ndarray): Right conservative state [u_R].

        Returns:
            np.ndarray: The numerical flux, calculated using the upwind method.
        """
        # For linear advection, HLLE simplifies to the upwind flux.
        a = self.speed
        if a >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)
