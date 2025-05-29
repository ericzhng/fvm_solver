import numpy as np
from .equation.base_equation import EquationSystem
from .limiter import Limiter


class Reconstruction:
    """Handles reconstruction methods for finite volume schemes.

    Supports piecewise constant, MUSCL, PPM, and WENO5 methods for interface state reconstruction.
    Ghost cells: n_ghost=2 suffices for MUSCL, WENO5, and simple PPM; n_ghost=3 preferred for full PPM.

    Attributes:
        equation_system (EquationSystem): System defining primitive/conservative conversions.
        limiter (Limiter, optional): Slope limiter for MUSCL reconstruction.
    """

    def __init__(self, equation_system: EquationSystem, limiter: str = None, limiter_beta: float = 1.5):
        """Initialize the reconstruction scheme.

        Args:
            equation_system (EquationSystem): System for state conversions.
            limiter (str, optional): Slope limiter type ('minmod', 'superbee', 'vanleer', etc.).
            limiter_beta (float, optional): Sharpness parameter for Osher/Sweby limiters (default: 1.5).

        Raises:
            TypeError: If equation_system is not an EquationSystem instance.
        """
        if not isinstance(equation_system, EquationSystem):
            raise TypeError("equation_system must be an EquationSystem instance")
        self.equation_system = equation_system
        self.limiter = Limiter(limiter, beta=limiter_beta) if limiter else None

    def _to_primitive_array(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).

        Returns:
            np.ndarray: Primitive variables, shape (n_vars, n_cells + 2*n_ghost).
        """
        n_cells = U.shape[1]
        W = np.empty_like(U)
        for i in range(n_cells):
            W[:, i] = self.equation_system.to_primitive(U[:, i])
        return W

    def _reconstruct_states(self, W_L: np.ndarray, W_R: np.ndarray, n_cells: int) -> tuple:
        """Convert primitive interface values to conservative states.

        Args:
            W_L (np.ndarray): Left primitive values, shape (n_vars, n_cells).
            W_R (np.ndarray): Right primitive values, shape (n_vars, n_cells).
            n_cells (int): Number of physical cells.

        Returns:
            tuple: (UL, UR), conservative states, each shape (n_vars, n_cells).
        """
        UL = np.array([self.equation_system.to_conservative(W_L[:, i]) for i in range(n_cells)]).T
        UR = np.array([self.equation_system.to_conservative(W_R[:, i]) for i in range(n_cells)]).T
        return UL, UR

    def _validate_input(self, U: np.ndarray, n_ghost: int):
        """Validate input array shape and ghost cells.

        Args:
            U (np.ndarray): Conservative variables.
            n_ghost (int): Number of ghost cells per side.

        Raises:
            ValueError: If U shape is invalid or insufficient ghost cells.
        """
        if U.ndim != 2 or U.shape[0] != self.equation_system.n_vars:
            raise ValueError(f"U must have shape (n_vars={self.equation_system.n_vars}, n_cells + 2*n_ghost)")
        if U.shape[1] < 2 * n_ghost + 1:
            raise ValueError("U must have at least 2*n_ghost + 1 cells")

    def piecewise_constant(self, U: np.ndarray, x: np.ndarray, n_ghost: int = 2, use_primitive: bool = False) -> tuple:
        """Perform piecewise constant reconstruction.

        UL[i] = U[i], UR[i] = U[i+1]. First-order accurate.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).
            x (np.ndarray): Spatial grid.
            n_ghost (int): Number of ghost cells per side (default: 2).

        Returns:
            tuple: (UL, UR), left and right states, each shape (n_vars, n_cells + 2).

        Raises:
            ValueError: If n_ghost < 1 or dx <= 0.
        """
        if n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        self._validate_input(U, n_ghost)

        if use_primitive:
            W = self._to_primitive_array(U)
            W_L = W[:, :-1]
            W_R = W[:, 1:]
            UL = np.array([self.equation_system.to_conservative(W_L[:, i]) for i in range(W_L.shape[1])]).T
            UR = np.array([self.equation_system.to_conservative(W_R[:, i]) for i in range(W_R.shape[1])]).T
        else:
            # use left and right side cell values for the interface
            UL = U[:, :-1]
            UR = U[:, 1:]

        return UL, UR

    def muscl(self, U: np.ndarray, x: np.ndarray, n_ghost: int = 2, use_primitive: bool = False) -> tuple:
        """Perform MUSCL reconstruction with slope limiting.

        Monotonic Upwind Scheme for Conservation Laws. Second-order accurate in smooth regions.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).
            x (np.ndarray): Spatial grid.
            n_ghost (int): Number of ghost cells per side (default: 2).

        Returns:
            tuple: (UL, UR), left and right states, each shape (n_vars, n_cells).

        Raises:
            ValueError: If n_ghost < 1 or dx <= 0.
        """
        if n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        self._validate_input(U, n_ghost)

        dx = np.diff(x)  # Non-uniform grid spacing
        n_cells_total = U.shape[1]

        if use_primitive:
            W = self._to_primitive_array(U)
            W_L = np.zeros_like(W[:, :-1])
            W_R = np.zeros_like(W[:, :-1])

            # Vectorized slope computation
            for j in range(self.equation_system.n_vars):
                left_slopes = (W[j, 1:-1] - W[j, :-2])/ dx[:-1]
                right_slopes = (W[j, 2:] - W[j, 1:-1])/ dx[1:]
                slopes = self.limiter.limit(left_slopes, right_slopes)

                slopes = np.insert(slopes, 0, 0)
                slopes = np.append(slopes, 0)

                W_L[j, :] = W[j, 0:-1] + 0.5 * dx[0:-1] * slopes[0:-1]
                W_R[j, :] = W[j, 1:] - 0.5 * dx[0:-1] * slopes[1:]

            return self._reconstruct_states(W_L, W_R, n_cells_total - 1)
        
        else:
            UL = np.zeros_like(U[:, :-1])
            UR = np.zeros_like(U[:, :-1])
            # Vectorized slope computation
            n_vars = U.shape[0]
            for j in range(n_vars):
                left_slopes = (U[j, 1:-1] - U[j, :-2]) / dx[:-1]
                right_slopes = (U[j, 2:] - U[j, 1:-1]) / dx[1:]
                slopes = self.limiter.limit(left_slopes, right_slopes)

                slopes = np.insert(slopes, 0, 0)
                slopes = np.append(slopes, 0)

                UL[j, :] = U[j, 0:-1] + 0.5 * dx[0:-1] * slopes[0:-1]
                UR[j, :] = U[j, 1:] - 0.5 * dx[0:-1] * slopes[1:]

            return UL, UR
