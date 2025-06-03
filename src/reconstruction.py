import numpy as np
from .equation.base_equation import EquationSystem
from .limiter import Limiter

class Reconstruction:
    """
    Handles reconstruction methods for finite volume schemes in 1D.

    Supports piecewise constant and MUSCL methods; extensible to PPM and WENO5.
    Maintains ghost cells in reconstructed states.

    Args:
        equation_system (EquationSystem): System for state conversions.
        limiter (str, optional): Slope limiter type ('minmod', 'superbee', 'vanleer', etc.).
        limiter_beta (float, optional): Sharpness parameter for Osher/Sweby limiters.
    """

    def __init__(self, equation_system: EquationSystem, reconstruct_in_primitive: bool = False, limiter: str = "", limiter_beta: float = 1.5):
        if not isinstance(equation_system, EquationSystem):
            raise TypeError("equation_system must be an EquationSystem instance")
        self.equation_system = equation_system
        self.limiter = Limiter(limiter, beta=limiter_beta) if limiter else None
        self.reconstruct_in_primitive = reconstruct_in_primitive

    def _validate_input(self, U: np.ndarray, n_ghost: int):
        """
        Validate input array shape and ghost cells.

        Args:
            U (np.ndarray): Conservative variables.
            n_ghost (int): Number of ghost cells per side.

        Raises:
            ValueError: If U shape is invalid or insufficient ghost cells.
        """
        if U.ndim < 2 or U.shape[0] != self.equation_system.num_vars:
            raise ValueError(f"U must have shape (num_vars={self.equation_system.num_vars}, n_cells + 2*n_ghost, ...)")
        if U.shape[1] < 2 * n_ghost + 1:
            raise ValueError("U must have at least 2*n_ghost + 1 cells")

    def piecewise_constant(self, U: np.ndarray, dx: np.ndarray, n_ghost: int = 2) -> tuple:
        """
        Piecewise constant reconstruction.

        Args:
            U (np.ndarray): Conservative variables, shape (num_vars, n_cells + 2*n_ghost, ...).
            dx (np.ndarray): Spatial grid distance array.
            n_ghost (int): Number of ghost cells per side.

        Returns:
            tuple: (UL, UR), left and right states, each shape (num_vars, n_cells + 2*n_ghost - 1, ...).
        """
        if n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        self._validate_input(U, n_ghost)

        # assign left and right states: at interface i, left state is U[i], right state is U[i+1]
        UL = U[:, :-1]
        UR = U[:, 1:]

        return UL, UR

    def muscl(self, U: np.ndarray, dx: np.ndarray, n_ghost: int = 2) -> tuple:
        """
        MUSCL reconstruction with slope limiting.

        Args:
            U (np.ndarray): Conservative variables, shape (num_vars, n_cells + 2*n_ghost, ...).
            dx (np.ndarray): Spatial grid distance array.
            n_ghost (int): Number of ghost cells per side.

        Returns:
            tuple: (UL, UR), left and right states, each shape (num_vars, n_cells + 2*n_ghost - 1, ...).
        """
        if n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        self._validate_input(U, n_ghost)

        if dx.ndim == 1:
            n_cells_total = U.shape[1]
            UL = np.zeros((self.equation_system.num_vars, n_cells_total - 1))
            UR = np.zeros((self.equation_system.num_vars, n_cells_total - 1))

            if self.reconstruct_in_primitive:
                W = self.equation_system.to_primitive_batch(U)
                for j in range(self.equation_system.num_vars):
                    left_slopes = (W[j, 1:-1] - W[j, :-2]) / dx[:-1]
                    right_slopes = (W[j, 2:] - W[j, 1:-1]) / dx[1:]
                    slopes = self.limiter.limit(left_slopes, right_slopes) if self.limiter else np.zeros_like(left_slopes)
                    slopes = np.insert(slopes, 0, 0)
                    slopes = np.append(slopes, 0)
                    W_L = W[j, :-1] + 0.5 * dx[:-1] * slopes[:-1]
                    W_R = W[j, 1:] - 0.5 * dx[:-1] * slopes[1:]
                    # Stack with other primitive variables for batch conversion
                    W_L_full = W.copy()
                    W_R_full = W.copy()
                    W_L_full[j, :-1] = W_L
                    W_R_full[j, 1:] = W_R
                    UL[j, :] = self.equation_system.to_conservative_batch(W_L_full[:, :-1])[j]
                    UR[j, :] = self.equation_system.to_conservative_batch(W_R_full[:, 1:])[j]
                return UL, UR
            else:
                for j in range(self.equation_system.num_vars):
                    dist = (dx[:-1] + dx[1:]) / 2
                    left_slopes = (U[j, 1:-1] - U[j, :-2]) / dist[:-1]
                    right_slopes = (U[j, 2:] - U[j, 1:-1]) / dist[1:]
                    slopes = self.limiter.limit(left_slopes, right_slopes) if self.limiter else np.zeros_like(left_slopes)
                    slopes = np.insert(slopes, 0, 0)
                    slopes = np.append(slopes, 0)
                    UL[j, :] = U[j, :-1] + 0.5 * dx[:-1] * slopes[:-1]
                    UR[j, :] = U[j, 1:] - 0.5 * dx[:-1] * slopes[1:]
                return UL, UR
        else:
            # Placeholder for 2D/3D MUSCL reconstruction
            raise NotImplementedError("2D/3D MUSCL reconstruction not yet implemented")