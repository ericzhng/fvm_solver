"""
This module defines the abstract base class `EqnBase` for hyperbolic
conservation law systems, providing a common interface for all equation-specific
implementations used in the FVM solver.
"""

import numpy as np
from abc import ABC, abstractmethod


class EqnBase(ABC):
    """
    Abstract base class for systems of hyperbolic conservation laws.

    This class defines the essential interface required by the FVM solver.
    Each specific equation system (e.g., Euler, Shallow Water) must inherit
    from this class and implement all its abstract methods.

    Attributes:
        min_value (float): A small positive value to prevent numerical issues
                         like division by zero, often used for density or
                         pressure floors.
        var_names (list[str]): A list of names for the primitive variables.
        num_vars (int): The number of variables in the equation system.
        vel_idx (int | None): The index of the velocity component in the state
                              vector. This is crucial for certain boundary
                              conditions (e.g., reflective).
    """

    def __init__(self, min_value: float = 1e-10):
        """
        Initializes the equation system base.

        Args:
            min_value (float, optional): A small positive value for numerical
                                       stability. Defaults to 1e-10.
        """
        self.min_value = min_value
        self.var_names: list[str] = []
        self.num_vars = 0
        self.vel_idx: int | None = None

    def get_var_names(self) -> list[str]:
        """
        Returns the names of the primitive variables.

        Returns:
            list[str]: A list of strings, where each string is the name of a
                       primitive variable.
        """
        return self.var_names

    @abstractmethod
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Converts a vector of primitive variables to conservative variables.

        Args:
            W (np.ndarray): A 1D NumPy array of primitive variables.
                            Shape: (num_vars,).

        Returns:
            np.ndarray: A 1D NumPy array of conservative variables.
                        Shape: (num_vars,).
        """
        pass

    @abstractmethod
    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Converts a vector of conservative variables to primitive variables.

        Args:
            U (np.ndarray): A 1D NumPy array of conservative variables.
                            Shape: (num_vars,).

        Returns:
            np.ndarray: A 1D NumPy array of primitive variables.
                        Shape: (num_vars,).
        """
        pass

    def to_conservative_batch(self, W: np.ndarray) -> np.ndarray:
        """
        Converts a batch of primitive variable vectors to conservative variables.

        This method applies the `to_conservative` conversion over a set of states.

        Args:
            W (np.ndarray): A 2D NumPy array of primitive variables, where each
                            column is a state vector. Shape: (num_vars, N).

        Returns:
            np.ndarray: A 2D NumPy array of conservative variables.
                        Shape: (num_vars, N).
        """
        # This implementation is a simple loop. For performance, subclasses
        # are encouraged to provide a vectorized implementation.
        return np.stack(
            [self.to_conservative(W[:, i]) for i in range(W.shape[1])], axis=1
        )

    def to_primitive_batch(self, U: np.ndarray) -> np.ndarray:
        """
        Converts a batch of conservative variable vectors to primitive variables.

        This method applies the `to_primitive` conversion over a set of states.

        Args:
            U (np.ndarray): A 2D NumPy array of conservative variables, where each
                            column is a state vector. Shape: (num_vars, N).

        Returns:
            np.ndarray: A 2D NumPy array of primitive variables.
                        Shape: (num_vars, N).
        """
        # This implementation is a simple loop. For performance, subclasses
        # are encouraged to provide a vectorized implementation.
        return np.stack([self.to_primitive(U[:, i]) for i in range(U.shape[1])], axis=1)

    @abstractmethod
    def max_eigenvalue(self, U: np.ndarray) -> float:
        """
        Computes the maximum absolute eigenvalue of the flux Jacobian matrix.

        This value, often related to the maximum wave speed, is critical for
        determining the stable time step size under the CFL condition.

        Args:
            U (np.ndarray): The conservative state vector. Shape: (num_vars,).

        Returns:
            float: The maximum absolute eigenvalue for the given state.
        """
        pass

    @abstractmethod
    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the physical flux vector F(U) for a given state.

        Args:
            U (np.ndarray): The conservative state vector. Shape: (num_vars,).

        Returns:
            np.ndarray: The physical flux vector. Shape: (num_vars,).
        """
        pass

    @abstractmethod
    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple:
        """
        Computes Roe-averaged quantities between two states.

        The specific quantities returned (e.g., averaged velocity, enthalpy)
        depend on the equation system and are used in the Roe flux calculation.

        Args:
            U_L (np.ndarray): The conservative state vector at the left interface.
            U_R (np.ndarray): The conservative state vector at the right interface.

        Returns:
            tuple: A tuple of Roe-averaged quantities.
        """
        pass

    # -------------------------------------------------------------------- #
    # Abstract methods for numerical flux functions                        #
    # These must be implemented by each specific equation class.           #
    # -------------------------------------------------------------------- #

    @abstractmethod
    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Roe numerical flux between two states.

        Args:
            U_L (np.ndarray): The conservative state vector at the left interface.
            U_R (np.ndarray): The conservative state vector at the right interface.

        Returns:
            np.ndarray: The Roe numerical flux vector.
        """
        pass

    @abstractmethod
    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the AUSM (Advection Upstream Splitting Method) numerical flux.

        Args:
            U_L (np.ndarray): The conservative state vector at the left interface.
            U_R (np.ndarray): The conservative state vector at the right interface.

        Returns:
            np.ndarray: The AUSM numerical flux vector.
        """
        pass

    @abstractmethod
    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLC (Harten-Lax-van Leer-Contact) numerical flux.

        Args:
            U_L (np.ndarray): The conservative state vector at the left interface.
            U_R (np.ndarray): The conservative state vector at the right interface.

        Returns:
            np.ndarray: The HLLC numerical flux vector.
        """
        pass

    @abstractmethod
    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the HLLE (Harten-Lax-van Leer-Einfeldt) numerical flux.

        Args:
            U_L (np.ndarray): The conservative state vector at the left interface.
            U_R (np.ndarray): The conservative state vector at the right interface.

        Returns:
            np.ndarray: The HLLE numerical flux vector.
        """
        pass
