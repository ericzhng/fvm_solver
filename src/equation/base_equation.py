import numpy as np
from abc import abstractmethod
from typing import Optional


class EquationSystem:
    """
    Abstract base class for hyperbolic conservation law systems.

    Provides interfaces for state conversion, flux calculation, and sound speed.
    Subclasses must implement all abstract methods for their specific system.

    Attributes:
        min_value (float): Minimum value for numerical stability.
        vel_idx (int): Index of the velocity variable in the state vector.
        monitor_idx (Optional[int]): Index of a variable to monitor for diagnostics.
        num_vars (int): Number of variables in the system.
        var_names (list[str]): Names of the primitive variables.
    """

    def __init__(self, min_value: float = 1e-10):
        """
        Initialize the equation system.

        Args:
            min_value (float): Minimum value for numerical stability.
        """
        self.min_value = min_value
        self.vel_idx: int = 0
        self.monitor_idx: Optional[int] = None
        self.num_vars = 0
        self.var_names: list[str] = []
        self.safety_guard_var_idx = []
        
    @abstractmethod
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Convert primitive variables to conservative variables.

        Args:
            W (np.ndarray): Primitive variable array, shape (num_vars,).

        Returns:
            np.ndarray: Conservative variable array, shape (num_vars,).
        """
        raise NotImplementedError

    @abstractmethod
    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variable array, shape (num_vars,).

        Returns:
            np.ndarray: Primitive variable array, shape (num_vars,).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute the physical flux for the given state.

        Args:
            U (np.ndarray): Conservative variables, shape (num_vars,).
            W (np.ndarray): Primitive variables, shape (num_vars,).

        Returns:
            np.ndarray: Flux vector, shape (num_vars,).
        """
        raise NotImplementedError

    @abstractmethod
    def sound_speed(self, W: np.ndarray) -> float:
        """
        Compute the sound speed for the given primitive state.

        Args:
            W (np.ndarray): Primitive variables, shape (num_vars,).

        Returns:
            float: Sound speed for the given state.
        """
        raise NotImplementedError

    @abstractmethod
    def hllc_numerical_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """
        Compute the HLLC numerical flux for the given left/right states.

        Args:
            WL (np.ndarray): Left primitive state, shape (num_vars,).
            WR (np.ndarray): Right primitive state, shape (num_vars,).
            UL (np.ndarray): Left conservative state, shape (num_vars,).
            UR (np.ndarray): Right conservative state, shape (num_vars,).

        Returns:
            np.ndarray: HLLC numerical flux, shape (num_vars,).

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("HLLC flux not implemented for this equation system.")

    @abstractmethod
    def roe_numerical_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """
        Compute the Roe numerical flux for the given left/right states.

        Args:
            WL (np.ndarray): Left primitive state, shape (num_vars,).
            WR (np.ndarray): Right primitive state, shape (num_vars,).
            UL (np.ndarray): Left conservative state, shape (num_vars,).
            UR (np.ndarray): Right conservative state, shape (num_vars,).

        Returns:
            np.ndarray: Roe numerical flux, shape (num_vars,).

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Roe flux not implemented for this equation system.")

    def get_variable_names(self) -> list:
        """
        Get the names of the primitive variables.

        Returns:
            list: List of variable names (str).
        """
        return self.var_names

    def to_conservative_batch(self, W: np.ndarray) -> np.ndarray:
        """
        Convert multiple columns of primitive variables to conservative variables.

        Args:
            W (np.ndarray): Primitive variables, shape (num_vars, N).

        Returns:
            np.ndarray: Conservative variables, shape (num_vars, N).
        """
        return np.stack([self.to_conservative(W[:, i]) for i in range(W.shape[1])], axis=1)

    def to_primitive_batch(self, U: np.ndarray) -> np.ndarray:
        """
        Convert multiple columns of conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables, shape (num_vars, N).

        Returns:
            np.ndarray: Primitive variables, shape (num_vars, N).
        """
        return np.stack([self.to_primitive(U[:, i]) for i in range(U.shape[1])], axis=1)
