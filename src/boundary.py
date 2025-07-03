import numpy as np
from typing import Optional
from .equation.equation_base import EqnBase


class BoundaryCondition:
    """
    Handles boundary conditions for finite volume schemes in 1D.

    This class supports Dirichlet, Neumann, periodic, and reflective boundary conditions.
    It expands the computational grid with ghost cells and populates them according to the specified boundary condition,
    ensuring accurate numerical solutions for hyperbolic PDEs.

    Attributes:
        eqn_obj (EquationBase): The equation system for state conversions.
        bc_kind (str): The type of boundary condition ('dirichlet', 'neumann', 'periodic', 'reflective').
        grid (np.ndarray): The spatial grid (1D array).
        left_boundary_state (np.ndarray): State or gradient at the left boundary.
        right_boundary_state (np.ndarray): State or gradient at the right boundary.
    """

    def __init__(
        self,
        eqn_obj: EqnBase,
        bc_kind: str,
        grid: np.ndarray,
        n_ghost: int,
        left_boundary_state: Optional[np.ndarray] = None,
        right_boundary_state: Optional[np.ndarray] = None,
    ):
        """
        Initialize the boundary condition handler.

        Args:
            eqn_obj (EquationBase): The system for conservative/primitive state conversions.
            bc_kind (str): Boundary condition type. One of {'dirichlet', 'neumann', 'periodic', 'reflective'}.
            grid (np.ndarray): 1D spatial grid.
            left_boundary_state (np.ndarray, optional): State or gradient at the left boundary. Defaults to zeros.
            right_boundary_state (np.ndarray, optional): State or gradient at the right boundary. Defaults to zeros.

        Raises:
            TypeError: If eqn_obj is not an EquationBase instance.
            ValueError: If bc_kind is not a supported type.
        """
        if not isinstance(eqn_obj, EqnBase):
            raise TypeError("eqn_obj must be an EquationBase instance")

        self.equation = eqn_obj
        self.bc_kind = bc_kind.lower()

        self.grid = grid
        self.n_cells = grid.size - 1
        self.n_ghost = n_ghost
        self.dx = np.diff(grid) if self.grid.size == 1 else np.diff(grid, axis=0)

        self.left_boundary_state = (
            left_boundary_state
            if left_boundary_state is not None
            else np.zeros(self.equation.num_vars)
        )
        self.right_boundary_state = (
            right_boundary_state
            if right_boundary_state is not None
            else np.zeros(self.equation.num_vars)
        )

        valid_bcs = {"dirichlet", "neumann", "periodic", "reflective"}
        if self.bc_kind not in valid_bcs:
            raise ValueError(f"bc_kind must be one of {valid_bcs}")

    def enforce_bc(self, U_aug: np.ndarray) -> np.ndarray:
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
        if self.n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        if U_aug.ndim < 2 or U_aug.shape[0] != self.equation.num_vars:
            raise ValueError(
                f"U must have shape ({self.equation.num_vars}, n_cells, ...)"
            )

        n_ghost = self.n_ghost
        n_cells = self.n_cells
        dx = self.dx
        U = U_aug[:, n_ghost : n_ghost + n_cells]

        # Populate ghost cells based on the boundary condition type
        if self.bc_kind == "dirichlet":
            U_aug[:, :n_ghost] = U[:, 1]
            U_aug[:, n_cells + n_ghost :] = U[:, 1]

        elif self.bc_kind == "neumann":
            W_left = self.equation.to_primitive(U[:, 0])
            W_right = self.equation.to_primitive(U[:, -1])
            for i in range(n_ghost):
                Wl = W_left - (n_ghost - i) * dx[0] * self.left_boundary_state
                U_aug[:, i] = self.equation.to_conservative(Wl)

                Wr = W_right + (i + 1) * dx[-1] * self.right_boundary_state
                U_aug[:, n_cells + n_ghost + i - 1] = self.equation.to_conservative(Wr)

        elif self.bc_kind == "periodic":
            U_aug[:, :n_ghost] = U_aug[:, n_cells : n_cells + n_ghost]
            U_aug[:, n_cells + n_ghost :] = U_aug[:, n_ghost : 2 * n_ghost]

        elif self.bc_kind == "reflective":
            if self.equation.vel_idx is None:
                raise ValueError("Reflective BC requires a valid vel_idx")

            W_left = self.equation.to_primitive(U[:, 0])
            W_right = self.equation.to_primitive(U[:, -1])
            for i in range(n_ghost):
                Wl = W_left.copy()
                Wl[self.equation.vel_idx] = -Wl[self.equation.vel_idx]
                U_aug[:, i] = self.equation.to_conservative(Wl)
                Wr = W_right.copy()
                Wr[self.equation.vel_idx] = -Wr[self.equation.vel_idx]
                U_aug[:, n_cells + n_ghost + i - 1] = self.equation.to_conservative(Wr)

        return U_aug
