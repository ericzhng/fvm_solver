import numpy as np
from typing import Optional
from .equation.base_equation import EquationSystem

class BoundaryCondition:
    """
    Handles boundary conditions for finite volume schemes in 1D.

    This class supports Dirichlet, Neumann, periodic, and reflective boundary conditions.
    It expands the computational grid with ghost cells and populates them according to the specified boundary condition,
    ensuring accurate numerical solutions for hyperbolic PDEs.

    Attributes:
        equation_system (EquationSystem): The equation system for state conversions.
        bc_kind (str): The type of boundary condition ('dirichlet', 'neumann', 'periodic', 'reflective').
        grid (np.ndarray): The spatial grid (1D array).
        left_boundary_state (np.ndarray): State or gradient at the left boundary.
        right_boundary_state (np.ndarray): State or gradient at the right boundary.
    """

    def __init__(
        self,
        equation_system: EquationSystem,
        bc_kind: str,
        grid: np.ndarray,
        left_boundary_state: Optional[np.ndarray] = None,
        right_boundary_state: Optional[np.ndarray] = None
    ):
        """
        Initialize the boundary condition handler.

        Args:
            equation_system (EquationSystem): The system for conservative/primitive state conversions.
            bc_kind (str): Boundary condition type. One of {'dirichlet', 'neumann', 'periodic', 'reflective'}.
            grid (np.ndarray): 1D spatial grid.
            left_boundary_state (np.ndarray, optional): State or gradient at the left boundary. Defaults to zeros.
            right_boundary_state (np.ndarray, optional): State or gradient at the right boundary. Defaults to zeros.

        Raises:
            TypeError: If equation_system is not an EquationSystem instance.
            ValueError: If bc_kind is not a supported type.
        """
        if not isinstance(equation_system, EquationSystem):
            raise TypeError("equation_system must be an EquationSystem instance")

        self.equation_system = equation_system
        self.bc_kind = bc_kind.lower()
        self.grid = grid
        self.left_boundary_state = left_boundary_state if left_boundary_state is not None else np.zeros(equation_system.num_vars)
        self.right_boundary_state = right_boundary_state if right_boundary_state is not None else np.zeros(equation_system.num_vars)
        
        valid_bcs = {'dirichlet', 'neumann', 'periodic', 'reflective'}
        if self.bc_kind not in valid_bcs:
            raise ValueError(f"bc_kind must be one of {valid_bcs}")

    def apply_bcs(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """
        Apply boundary conditions to the solution array, expanding it with ghost cells.

        Args:
            U (np.ndarray): Solution array, conservative variables (U), shape (n_vars, n_cells, ...).
            n_ghost (int): Number of ghost cells to add on each side of the domain.

        Returns:
            np.ndarray: Solution array with ghost cells populated, shape (n_vars, n_cells + 2 * n_ghost, ...).

        Raises:
            ValueError: If n_ghost < 1 or if U has an invalid shape.
            NotImplementedError: If called for 2D/3D arrays (only 1D is implemented).
        """
        if n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        if U.ndim < 2 or U.shape[0] != self.equation_system.num_vars:
            raise ValueError(f"U must have shape ({self.equation_system.num_vars}, n_cells, ...)")
        n_cells = U.shape[1]
        U_work = U.copy()

        if U.ndim == 2:  # 1D
            n_cells_total = n_cells + 2 * n_ghost
            U_new = np.zeros([self.equation_system.num_vars, n_cells_total])
            U_new[:, n_ghost:n_ghost + n_cells] = U_work

            grid_dx = self.grid[1:] - self.grid[0:-1]
            grid_dx_new = np.zeros(n_cells_total)
            grid_dx_new[n_ghost:n_ghost + n_cells] = grid_dx
            grid_dx_new[:n_ghost] = grid_dx[0]
            grid_dx_new[n_cells + n_ghost:] = grid_dx[-1]

            if self.bc_kind == 'dirichlet':
                U_left = self.equation_system.to_conservative(self.left_boundary_state)
                U_right = self.equation_system.to_conservative(self.right_boundary_state)
                U_new[:, :n_ghost] = U_left[:, np.newaxis]
                U_new[:, n_cells + n_ghost:] = U_right[:, np.newaxis]

            elif self.bc_kind == 'neumann':
                W_left = self.equation_system.to_primitive(U_work[:, 0])
                W_right = self.equation_system.to_primitive(U_work[:, -1])
                for i in range(n_ghost):
                    Wl = W_left - (n_ghost - i) * grid_dx_new[0] * self.left_boundary_state
                    U_new[:, i] = self.equation_system.to_conservative(Wl)

                    Wr = W_right + (i + 1) * grid_dx_new[-1] * self.right_boundary_state
                    U_new[:, n_cells + n_ghost + i] = self.equation_system.to_conservative(Wr)

            elif self.bc_kind == 'periodic':
                U_new[:, :n_ghost] = U_new[:, n_cells:n_cells + n_ghost]
                U_new[:, n_cells + n_ghost:] = U_new[:, n_ghost:2 * n_ghost]

            elif self.bc_kind == 'reflective':
                if self.equation_system.vel_idx is None:
                    raise ValueError("Reflective BC requires a valid vel_idx")
                
                W_left = self.equation_system.to_primitive(U_work[:, 0])
                W_right = self.equation_system.to_primitive(U_work[:, -1])
                for i in range(n_ghost):
                    Wl = W_left.copy()
                    Wl[self.equation_system.vel_idx] = -Wl[self.equation_system.vel_idx]
                    U_new[:, i] = self.equation_system.to_conservative(Wl)
                    Wr = W_right.copy()
                    Wr[self.equation_system.vel_idx] = -Wr[self.equation_system.vel_idx]
                    U_new[:, n_cells + n_ghost + i] = self.equation_system.to_conservative(Wr)

            return U_new
        else:
            # Placeholder for 2D/3D boundary conditions
            raise NotImplementedError("2D/3D boundary conditions not yet implemented")
