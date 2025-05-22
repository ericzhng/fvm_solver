import numpy as np
from .equation import EquationSystem
from .limiters import Limiter


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
            W_L (np.ndarray): Left primitive values, shape (n_vars, n_cells - 1).
            W_R (np.ndarray): Right primitive values, shape (n_vars, n_cells - 1).
            n_cells (int): Number of physical cells.

        Returns:
            tuple: (UL, UR), conservative states, each shape (n_vars, n_cells - 1).
        """
        UL = np.array([self.equation_system.to_conservative(W_L[:, i]) for i in range(n_cells - 1)]).T
        UR = np.array([self.equation_system.to_conservative(W_R[:, i]) for i in range(n_cells - 1)]).T
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

    def piecewise_constant(self, U: np.ndarray, dx: float, n_ghost: int = 2) -> tuple:
        """Perform piecewise constant reconstruction.

        UL[i] = U[i], UR[i] = U[i+1]. First-order accurate.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).
            dx (float): Spatial grid spacing.
            n_ghost (int): Number of ghost cells per side (default: 2).

        Returns:
            tuple: (UL, UR), left and right states, each shape (n_vars, n_cells - 1).

        Raises:
            ValueError: If n_ghost < 1 or dx <= 0.
        """
        if n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        if dx <= 0:
            raise ValueError("dx must be positive")
        self._validate_input(U, n_ghost)
        n_vars, n_cells_total = U.shape
        n_cells = n_cells_total - 2 * n_ghost
        UL = U[:, n_ghost:n_cells + n_ghost]
        UR = U[:, n_ghost + 1:n_cells + n_ghost + 1]
        return UL, UR

    def muscl(self, U: np.ndarray, dx: float, n_ghost: int = 2) -> tuple:
        """Perform MUSCL reconstruction with slope limiting.

        Monotonic Upwind Scheme for Conservation Laws. Second-order accurate in smooth regions.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).
            dx (float): Spatial grid spacing.
            n_ghost (int): Number of ghost cells per side (default: 2).

        Returns:
            tuple: (UL, UR), left and right states, each shape (n_vars, n_cells - 1).

        Raises:
            ValueError: If n_ghost < 1 or dx <= 0.
        """
        if n_ghost < 1:
            raise ValueError("n_ghost must be at least 1")
        if dx <= 0:
            raise ValueError("dx must be positive")
        self._validate_input(U, n_ghost)
        n_vars, n_cells_total = U.shape
        n_cells = n_cells_total - 2 * n_ghost
        W = self._to_primitive_array(U)
        W_L = np.zeros((n_vars, n_cells - 1))
        W_R = np.zeros((n_vars, n_cells - 1))

        # Vectorized slope computation
        for j in range(n_vars):
            sigma_L = np.zeros(n_cells - 1)
            sigma_R = np.zeros(n_cells - 1)
            if self.limiter:
                left_slopes = (W[j, n_ghost:n_cells + n_ghost] - W[j, n_ghost - 1:n_cells + n_ghost - 1]) / dx
                right_slopes = (W[j, n_ghost + 1:n_cells + n_ghost + 1] - W[j, n_ghost:n_cells + n_ghost]) / dx
                sigma_L = self.limiter.limit(left_slopes, right_slopes)
                sigma_R = self.limiter.limit(right_slopes, (W[j, n_ghost + 2:n_cells + n_ghost + 2] - W[j, n_ghost + 1:n_cells + n_ghost + 1]) / dx)
            else:
                sigma_L = (W[j, n_ghost:n_cells + n_ghost] - W[j, n_ghost - 1:n_cells + n_ghost - 1]) / dx
                sigma_R = (W[j, n_ghost + 1:n_cells + n_ghost + 1] - W[j, n_ghost:n_cells + n_ghost]) / dx
            W_L[j, :] = W[j, n_ghost:n_cells + n_ghost] + 0.5 * dx * sigma_L
            W_R[j, :] = W[j, n_ghost + 1:n_cells + n_ghost + 1] - 0.5 * dx * sigma_R

        return self._reconstruct_states(W_L, W_R, n_cells)

    def ppm(self, U: np.ndarray, dx: float, n_ghost: int = 2, full_stencil: bool = False) -> tuple:
        """Perform Piecewise Parabolic Method (PPM) reconstruction.

        Constructs parabolic profiles in each cell, applies monotonicity constraints.
        Third-order accurate in smooth regions with full stencil.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).
            dx (float): Grid spacing.
            n_ghost (int): Number of ghost cells per side (default: 2).
            full_stencil (bool): Use 5-point stencil for higher accuracy (requires n_ghost >= 2).

        Returns:
            tuple: (UL, UR), left and right states, each shape (n_vars, n_cells - 1).

        Raises:
            ValueError: If n_ghost is insufficient or dx <= 0.
        """
        if n_ghost < (2 if full_stencil else 1):
            raise ValueError(f"n_ghost must be at least {2 if full_stencil else 1}")
        if dx <= 0:
            raise ValueError("dx must be positive")
        self._validate_input(U, n_ghost)
        n_vars, n_cells_total = U.shape
        n_cells = n_cells_total - 2 * n_ghost
        W = self._to_primitive_array(U)
        W_L_all = np.zeros((n_vars, n_cells))
        W_R_all = np.zeros((n_vars, n_cells))

        for j in range(n_vars):
            if full_stencil:
                # 5-point stencil for interior cells
                idx = slice(n_ghost + 2, n_cells + n_ghost - 2)
                W_R_all[j, 2:-2] = (7/12) * (W[j, idx] + W[j, idx + 1]) - (1/12) * (W[j, idx - 1] + W[j, idx + 2])
                W_L_all[j, 2:-2] = (7/12) * (W[j, idx - 1] + W[j, idx]) - (1/12) * (W[j, idx - 2] + W[j, idx + 1])
                # Simple stencil for boundary cells
                idx_bound = [0, 1, n_cells - 2, n_cells - 1]
                W_R_all[j, idx_bound] = W[j, n_ghost + idx_bound] + 0.5 * (W[j, n_ghost + idx_bound + 1] - W[j, n_ghost + idx_bound - 1])
                W_L_all[j, idx_bound] = W[j, n_ghost + idx_bound] - 0.5 * (W[j, n_ghost + idx_bound + 1] - W[j, n_ghost + idx_bound - 1])
            else:
                # Simple second-order stencil
                delta_W = 0.5 * (W[j, n_ghost + 1:n_cells + n_ghost + 1] - W[j, n_ghost - 1:n_cells + n_ghost - 1])
                W_R_all[j, :] = W[j, n_ghost:n_cells + n_ghost] + delta_W
                W_L_all[j, :] = W[j, n_ghost:n_cells + n_ghost] - delta_W

            # Apply monotonicity constraints
            for i in range(n_cells):
                if (W_R_all[j, i] - W[j, i + n_ghost]) * (W[j, i + n_ghost] - W_L_all[j, i]) <= 0:
                    W_L_all[j, i] = W[j, i + n_ghost]
                    W_R_all[j, i] = W[j, i + n_ghost]
                else:
                    delta_W = W_R_all[j, i] - W_L_all[j, i] + 1e-10
                    if (W_R_all[j, i] - W[j, i + n_ghost]) * (W[j, i + n_ghost] - W_L_all[j, i]) > delta_W**2 / 6:
                        W_L_all[j, i] = 3 * W[j, i + n_ghost] - 2 * W_R_all[j, i]
                    if (W_R_all[j, i] - W[j, i + n_ghost]) * (W[j, i + n_ghost] - W_L_all[j, i]) < -delta_W**2 / 6:
                        W_R_all[j, i] = 3 * W[j, i + n_ghost] - 2 * W_L_all[j, i]

        W_L = W_R_all[:, :-1]
        W_R = W_L_all[:, 1:]
        return self._reconstruct_states(W_L, W_R, n_cells)

    def weno5(self, U: np.ndarray, dx: float, n_ghost: int = 2) -> tuple:
        """Perform WENO5 reconstruction.

        Uses weighted combination of three stencils for fifth-order accuracy in smooth regions.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).
            dx (float): Spatial grid spacing.
            n_ghost (int): Number of ghost cells per side (default: 2).

        Returns:
            tuple: (UL, UR), left and right states, each shape (n_vars, n_cells - 1).

        Raises:
            ValueError: If n_ghost < 2 or dx <= 0.
        """
        if n_ghost < 2:
            raise ValueError("n_ghost must be at least 2 for WENO5")
        if dx <= 0:
            raise ValueError("dx must be positive")
        self._validate_input(U, n_ghost)
        n_vars, n_cells_total = U.shape
        n_cells = n_cells_total - 2 * n_ghost
        W = self._to_primitive_array(U)
        W_L = np.zeros((n_vars, n_cells - 1))
        W_R = np.zeros((n_vars, n_cells - 1))
        epsilon = 1e-6 * np.max(np.abs(W)) + 1e-10  # Dynamic epsilon for stability

        for i in range(n_cells - 1):
            idx = i + n_ghost
            for j in range(n_vars):
                v = W[j, idx - 2:idx + 3]
                # Smoothness indicators
                beta0 = 13.0 / 12.0 * (v[0] - 2 * v[1] + v[2])**2 + 0.25 * (v[0] - 4 * v[1] + 3 * v[2])**2
                beta1 = 13.0 / 12.0 * (v[1] - 2 * v[2] + v[3])**2 + 0.25 * (v[1] - v[3])**2
                beta2 = 13.0 / 12.0 * (v[2] - 2 * v[3] + v[4])**2 + 0.25 * (3 * v[2] - 4 * v[3] + v[4])**2
                # Weights
                d0, d1, d2 = 0.1, 0.6, 0.3
                alpha0 = d0 / (beta0 + epsilon)**2
                alpha1 = d1 / (beta1 + epsilon)**2
                alpha2 = d2 / (beta2 + epsilon)**2
                sum_alpha = alpha0 + alpha1 + alpha2
                omega0 = alpha0 / sum_alpha
                omega1 = alpha1 / sum_alpha
                omega2 = alpha2 / sum_alpha
                # Left state polynomials
                p0 = (2 * v[0] - 7 * v[1] + 11 * v[2]) / 6.0
                p1 = (-v[1] + 5 * v[2] + 2 * v[3]) / 6.0
                p2 = (2 * v[2] + 5 * v[3] - v[4]) / 6.0
                W_L[j, i] = omega0 * p0 + omega1 * p1 + omega2 * p2
                # Right state polynomials
                p0 = (-v[0] + 5 * v[1] + 2 * v[2]) / 6.0
                p1 = (2 * v[1] + 5 * v[2] - v[3]) / 6.0
                p2 = (11 * v[2] - 7 * v[3] + 2 * v[4]) / 6.0
                W_R[j, i] = omega0 * p0 + omega1 * p1 + omega2 * p2

        return self._reconstruct_states(W_L, W_R, n_cells)
