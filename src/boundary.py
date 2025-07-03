"""
This module defines the BoundaryCondition class for handling boundary conditions
in a 1D finite volume method (FVM) solver.
"""

import numpy as np
from typing import Optional
from .equation.equation_base import EqnBase


class BoundaryCondition:
    """
    Handles boundary conditions for 1D finite volume schemes.

    This class applies various types of boundary conditions by populating
    ghost cells at the boundaries of the computational domain. It supports
    Dirichlet, Neumann, periodic, and reflective boundary conditions, which are
    essential for solving hyperbolic partial differential equations accurately.

    Attributes:
        equation (EqnBase): The equation system object, providing methods for
                             state conversions (e.g., conservative to primitive).
        bc_kind (str): The type of boundary condition to be applied.
                       Supported types: 'dirichlet', 'neumann', 'periodic',
                       'reflective'.
        grid (np.ndarray): The 1D spatial grid.
        n_cells (int): The number of computational cells in the grid.
        n_ghost (int): The number of ghost cells on each side of the domain.
        dx (np.ndarray): The grid spacing for each cell.
        left_boundary_state (np.ndarray): The state or gradient value at the
                                          left boundary. Used for Dirichlet or
                                          Neumann conditions.
        right_boundary_state (np.ndarray): The state or gradient value at the
                                           right boundary. Used for Dirichlet or
                                           Neumann conditions.
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
        Initializes the BoundaryCondition handler.

        Args:
            eqn_obj (EqnBase): An instance of an equation system class
                               (e.g., EquationEuler), which provides the
                               necessary physics-specific transformations.
            bc_kind (str): The type of boundary condition. Must be one of
                           {'dirichlet', 'neumann', 'periodic', 'reflective'}.
            grid (np.ndarray): The 1D spatial grid defining the cell centers.
            n_ghost (int): The number of ghost cells to add to each side of
                           the computational domain.
            left_boundary_state (np.ndarray, optional): The state (for
                Dirichlet) or gradient (for Neumann) at the left boundary.
                Defaults to a zero vector if not provided.
            right_boundary_state (np.ndarray, optional): The state (for
                Dirichlet) or gradient (for Neumann) at the right boundary.
                Defaults to a zero vector if not provided.

        Raises:
            TypeError: If `eqn_obj` is not an instance of `EqnBase`.
            ValueError: If `bc_kind` is not a supported boundary condition type.
        """
        if not isinstance(eqn_obj, EqnBase):
            raise TypeError("eqn_obj must be an instance of EqnBase.")

        self.equation = eqn_obj
        self.bc_kind = bc_kind.lower()

        self.grid = grid
        self.n_cells = grid.size - 1
        self.n_ghost = n_ghost
        self.dx = np.diff(grid) if self.grid.size == 1 else np.diff(grid, axis=0)

        # Set default boundary states if not provided
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

        # Validate the boundary condition type
        valid_bcs = {"dirichlet", "neumann", "periodic", "reflective"}
        if self.bc_kind not in valid_bcs:
            raise ValueError(f"bc_kind must be one of {valid_bcs}.")

    def enforce_bc(self, U_aug: np.ndarray) -> np.ndarray:
        """
        Applies the specified boundary conditions to the solution array.

        This method populates the ghost cells of an augmented solution array
        (`U_aug`) based on the chosen boundary condition type.

        Args:
            U_aug (np.ndarray): The augmented solution array of conservative
                                variables, including space for ghost cells.
                                Shape: (num_vars, n_cells + 2 * n_ghost).

        Returns:
            np.ndarray: The solution array with ghost cells correctly populated.
                        Shape remains (num_vars, n_cells + 2 * n_ghost).

        Raises:
            ValueError: If `n_ghost` < 1, or if `U_aug` has an invalid shape.
            NotImplementedError: If reflective boundary conditions are used
                                 for an equation system without a defined
                                 velocity index (`vel_idx`).
        """
        if self.n_ghost < 1:
            raise ValueError("n_ghost must be at least 1.")
        if U_aug.ndim != 2 or U_aug.shape[0] != self.equation.num_vars:
            raise ValueError(
                f"U_aug must have shape ({self.equation.num_vars}, "
                f"n_cells + 2 * n_ghost)."
            )

        n_ghost = self.n_ghost
        n_cells = self.n_cells
        dx = self.dx

        # Extract the interior solution to be used for setting ghost cells
        U = U_aug[:, n_ghost : n_ghost + n_cells]

        # Populate ghost cells based on the boundary condition type
        if self.bc_kind == "dirichlet":
            # Fixed value boundary condition
            # The ghost cells are set to the prescribed boundary state.
            U_aug[:, :n_ghost] = U[:, 1]
            U_aug[:, n_cells + n_ghost :] = U[:, 1]

        elif self.bc_kind == "neumann":
            # Fixed gradient (zero-order extrapolation) boundary condition
            # The ghost cell values are extrapolated from the interior.
            W_left = self.equation.to_primitive(U[:, 0])
            W_right = self.equation.to_primitive(U[:, -1])
            for i in range(n_ghost):
                # Extrapolate to the left ghost cells
                Wl = W_left - (n_ghost - i) * dx[0] * self.left_boundary_state
                U_aug[:, i] = self.equation.to_conservative(Wl)

                # Extrapolate to the right ghost cells
                Wr = W_right + (i + 1) * dx[-1] * self.right_boundary_state
                U_aug[:, n_cells + n_ghost + i - 1] = self.equation.to_conservative(Wr)

        elif self.bc_kind == "periodic":
            # Periodic boundary condition
            # The ghost cells on one side are filled with values from the
            # other side of the domain.
            U_aug[:, :n_ghost] = U_aug[:, n_cells : n_cells + n_ghost]
            U_aug[:, n_cells + n_ghost :] = U_aug[:, n_ghost : 2 * n_ghost]

        elif self.bc_kind == "reflective":
            # Reflective (solid wall) boundary condition
            # This condition reflects the flow, typically by negating the
            # normal velocity component at the boundary.
            if self.equation.vel_idx is None:
                raise NotImplementedError(
                    "Reflective BC requires a valid vel_idx in the equation."
                )

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
