import numpy as np
from src.equation import EquationSystem

class BoundaryCondition:
    """Handles boundary conditions for finite volume schemes.

    Supports Dirichlet, Neumann, periodic, and reflective boundary conditions.
    Applies conditions to ghost cells to ensure accurate numerical solutions.

    Attributes:
        equation_system (EquationSystem): System for state conversions.
        bc_type (str): Boundary condition type ('dirichlet', 'neumann', 'periodic', 'reflective').
        left_values (np.ndarray): Values or gradients at left boundary.
        right_values (np.ndarray): Values or gradients at right boundary.
        dx (float): Grid spacing for Neumann conditions.
    """

    def __init__(self, equation_system: EquationSystem, bc_type: str, 
                 left_values: np.ndarray = None, right_values: np.ndarray = None, dx: float = 1.0):
        """Initialize boundary condition handler.

        Args:
            equation_system (EquationSystem): System for primitive/conservative conversions.
            bc_type (str): Type of boundary condition ('dirichlet', 'neumann', 'periodic', 'reflective').
            left_values (np.ndarray, optional): Dirichlet values or Neumann gradients at left boundary.
            right_values (np.ndarray, optional): Dirichlet values or Neumann gradients at right boundary.
            dx (float, optional): Grid spacing for Neumann conditions. Must be positive.

        Raises:
            TypeError: If equation_system is not an EquationSystem instance.
            ValueError: If bc_type is unsupported or dx is non-positive.
        """
        if not isinstance(equation_system, EquationSystem):
            raise TypeError("equation_system must be an EquationSystem instance")
        if dx <= 0:
            raise ValueError("dx must be positive")
        self.equation_system = equation_system
        self.bc_type = bc_type.lower()
        self.left_values = (left_values if left_values is not None 
                          else np.zeros(equation_system.n_vars))
        self.right_values = (right_values if right_values is not None 
                           else np.zeros(equation_system.n_vars))
        self.dx = dx
        valid_bcs = {'dirichlet', 'neumann', 'periodic', 'reflective'}
        if self.bc_type not in valid_bcs:
            raise ValueError(f"bc_type must be one of {valid_bcs}")

    def apply_bcs(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply boundary conditions to conservative variables.

        Populates ghost cells based on the specified boundary condition type.
        - Dirichlet: Fixes primitive values at boundaries.
        - Neumann: Enforces specified gradients in primitive variables.
        - Periodic: Copies states from opposite boundaries.
        - Reflective: Mirrors states, negating velocity for physical reflection.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).
            n_ghost (int): Number of ghost cells per side.

        Returns:
            np.ndarray: Updated array with ghost cells populated.

        Raises:
            ValueError: If n_ghost < 1 or input shape is invalid.
        """
        if n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        if U.ndim != 2 or U.shape[0] != self.equation_system.n_vars:
            raise ValueError(f"U must have shape ({self.equation_system.n_vars}, n_cells + 2*n_ghost)")

        n_vars, n_cells_total = U.shape
        n_cells = n_cells_total - 2 * n_ghost
        U_new = np.copy(U)  # Copy to avoid modifying input

        if self.bc_type == 'dirichlet':
            # Set ghost cells to fixed primitive values, convert to conservative
            left_conservative = self.equation_system.to_conservative(self.left_values)
            right_conservative = self.equation_system.to_conservative(self.right_values)
            U_new[:, :n_ghost] = left_conservative[:, np.newaxis]
            U_new[:, n_cells + n_ghost:] = right_conservative[:, np.newaxis]

        elif self.bc_type == 'neumann':
            # Enforce gradient in primitive variables
            W = self.equation_system.to_primitive(U.T).T  # Convert all cells to primitive
            for i in range(n_ghost):
                # Left: W_i = W_n_ghost - (n_ghost - i) * dx * gradient
                W_left = W[:, n_ghost] - (n_ghost - i) * self.dx * self.left_values
                U_new[:, i] = self.equation_system.to_conservative(W_left)
                # Right: W_{n_cells+n_ghost+i} = W_{n_cells+n_ghost-1} + (i + 1) * dx * gradient
                W_right = W[:, n_cells + n_ghost - 1] + (i + 1) * self.dx * self.right_values
                U_new[:, n_cells + n_ghost + i] = self.equation_system.to_conservative(W_right)

        elif self.bc_type == 'periodic':
            # Copy physical cells from opposite boundary
            U_new[:, :n_ghost] = U_new[:, n_cells:n_cells + n_ghost]
            U_new[:, n_cells + n_ghost:] = U_new[:, n_ghost:2 * n_ghost]

        elif self.bc_type == 'reflective':
            # Reflect velocity, copy other variables
            if self.equation_system.velocity_index is None:
                raise ValueError("Reflective BC requires a valid velocity_index")
            W = self.equation_system.to_primitive(U.T).T
            for i in range(n_ghost):
                # Left: Mirror state, negate velocity
                W_left = W[:, n_ghost].copy()
                W_left[self.equation_system.velocity_index] = -W_left[self.equation_system.velocity_index]
                U_new[:, i] = self.equation_system.to_conservative(W_left)
                # Right: Mirror state, negate velocity
                W_right = W[:, n_cells + n_ghost - 1].copy()
                W_right[self.equation_system.velocity_index] = -W_right[self.equation_system.velocity_index]
                U_new[:, n_cells + n_ghost + i] = self.equation_system.to_conservative(W_right)

        return U_new
