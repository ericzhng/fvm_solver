import numpy as np


class EquationSystem:
    """Base class for hyperbolic conservation law equation systems.

    Provides default numerical methods for flux calculations and state conversions.
    Subclasses should override methods for analytical implementations.
    """

    def __init__(self, min_var: float = 1e-10):
        """Initialize equation system with minimum variable threshold."""
        self.min_var = min_var  # Minimum value for numerical stability

        self.velocity_index = None
        self.monitored_index = None
        self.safeguarded_indices: list[int] = [] # for primitive variables

        self.n_vars = 0
        self.variable_names: list[str] = []
    
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Args:
            W (np.ndarray): Primitive variables.

        Returns:
            np.ndarray: Conservative variables.
        """
        raise NotImplementedError
    
    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables.

        Returns:
            np.ndarray: Primitive variables.
        """
        raise NotImplementedError

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute the physical flux for the given state.

        Args:
            U (np.ndarray): Conservative variables.
            W (np.ndarray): Primitive variables.

        Returns:
            np.ndarray: Flux vector.
        """
        raise NotImplementedError

    def sound_speed(self, W: np.ndarray) -> float:
        """Compute the sound speed for the given primitive state.

        Args:
            W (np.ndarray): Primitive variables.

        Returns:
            float: Sound speed.
        """
        raise NotImplementedError

    def get_variable_names(self) -> list:
        """Return the names of the primitive variables.

        Returns:
            list: List of variable names.
        """
        return self.variable_names
