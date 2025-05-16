import numpy as np
from .equation import EquationSystem
from .limiters import Limiter


class Reconstruction:
    """Class to perform spatial reconstructions for Godunov-type schemes.

    Supports piecewise constant, MUSCL, PPM, and WENO5 methods.
    """

    def __init__(self, equation_system: EquationSystem, limiter: str = None):
        """Initialize the reconstruction scheme.

        Args:
            equation_system (EquationSystem): The equation system for state conversions.
            limiter (str, optional): Slope limiter for MUSCL ('minmod', 'superbee', 'vanleer', 'none').
        """
        self.equation_system = equation_system
        self.limiter = Limiter(limiter) if limiter else None

    def piecewise_constant(self, U: np.ndarray, dx: float) -> tuple:
        """Perform piecewise constant reconstruction.

        UL[i] = U[i], UR[i] = U[i+1]

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            dx (float): Spatial grid spacing.

        Returns:
            tuple: Left and right reconstructed states (UL, UR).
        """
        n_vars, n_cells = U.shape
        UL = U[:, :-1].copy()
        UR = U[:, 1:].copy()
        return UL, UR

    def muscl(self, U: np.ndarray, dx: float) -> tuple:
        """Perform MUSCL reconstruction with slope limiting.

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            dx (float): Spatial grid spacing.

        Returns:
            tuple: Left and right reconstructed states (UL, UR).
        """
        n_vars, n_cells = U.shape
        UL = np.zeros((n_vars, n_cells - 1))
        UR = np.zeros((n_vars, n_cells - 1))
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n_cells)]).T
        for i in range(2, n_cells - 2):
            for j in range(n_vars):
                # Compute slopes
                sigma_L = self.limiter.limit(
                    (W[j, i] - W[j, i - 1]) / dx,
                    (W[j, i - 1] - W[j, i - 2]) / dx
                ) if self.limiter else (W[j, i] - W[j, i - 1]) / dx
                sigma_R = self.limiter.limit(
                    (W[j, i + 1] - W[j, i]) / dx,
                    (W[j, i] - W[j, i - 1]) / dx
                ) if self.limiter else (W[j, i + 1] - W[j, i]) / dx
                # Reconstruct primitive variables
                W_L = W[j, i] - 0.5 * dx * sigma_L
                W_R = W[j, i] + 0.5 * dx * sigma_R
                # Convert back to conservative
                W_tmp_L = W[:, i].copy()
                W_tmp_R = W[:, i].copy()
                W_tmp_L[j] = W_L
                W_tmp_R[j] = W_R
                UL[:, i - 1] = self.equation_system.to_conservative(W_tmp_L)
                UR[:, i - 1] = self.equation_system.to_conservative(W_tmp_R)
        return UL, UR

    def ppm(self, U: np.ndarray, dx: float) -> tuple:
        """Perform Piecewise Parabolic Method (PPM) reconstruction.

        Simplified version: averages adjacent states.

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            dx (float): Spatial grid spacing.

        Returns:
            tuple: Left and right reconstructed states (UL, UR).
        """
        n_vars, n_cells = U.shape
        UL = np.zeros((n_vars, n_cells - 1))
        UR = np.zeros((n_vars, n_cells - 1))
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n_cells)]).T
        for i in range(2, n_cells - 2):
            for j in range(n_vars):
                # Simplified PPM: average of adjacent cells
                W_L = 0.5 * (W[j, i] + W[j, i - 1])
                W_R = 0.5 * (W[j, i + 1] + W[j, i])
                W_tmp_L = W[:, i].copy()
                W_tmp_R = W[:, i].copy()
                W_tmp_L[j] = W_L
                W_tmp_R[j] = W_R
                UL[:, i - 1] = self.equation_system.to_conservative(W_tmp_L)
                UR[:, i - 1] = self.equation_system.to_conservative(W_tmp_R)
        return UL, UR

    def weno5(self, U: np.ndarray, dx: float) -> tuple:
        """Perform WENO5 reconstruction.

        Uses weighted combination of three stencils for high-order accuracy.

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            dx (float): Spatial grid spacing.

        Returns:
            tuple: Left and right reconstructed states (UL, UR).
        """
        n_vars, n_cells = U.shape
        UL = np.zeros((n_vars, n_cells - 1))
        UR = np.zeros((n_vars, n_cells - 1))
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n_cells)]).T
        epsilon = max(1e-6, 1e-10 * np.max(np.abs(W)))  # Regularization parameter
        for i in range(2, n_cells - 2):
            for j in range(n_vars):
                v = W[j, i - 2:i + 3]
                # Smoothness indicators
                beta0 = (13.0 / 12.0) * (v[0] - 2 * v[1] + v[2])**2 + 0.25 * (v[0] - 4 * v[1] + 3 * v[2])**2
                beta1 = (13.0 / 12.0) * (v[1] - 2 * v[2] + v[3])**2 + 0.25 * (v[1] - v[3])**2
                beta2 = (13.0 / 12.0) * (v[2] - 2 * v[3] + v[4])**2 + 0.25 * (3 * v[2] - 4 * v[3] + v[4])**2
                # Nonlinear weights
                d0, d1, d2 = 0.1, 0.6, 0.3  # Ideal weights
                alpha0 = d0 / (beta0 + epsilon)**2
                alpha1 = d1 / (beta1 + epsilon)**2
                alpha2 = d2 / (beta2 + epsilon)**2
                sum_alpha = alpha0 + alpha1 + alpha2
                omega0 = alpha0 / sum_alpha
                omega1 = alpha1 / sum_alpha
                omega2 = alpha2 / sum_alpha
                # Polynomial reconstructions
                p0 = (2 * v[0] - 7 * v[1] + 11 * v[2]) / 6.0
                p1 = (-v[1] + 5 * v[2] + 2 * v[3]) / 6.0
                p2 = (2 * v[2] + 5 * v[3] - v[4]) / 6.0
                W_L = omega0 * p0 + omega1 * p1 + omega2 * p2
                # Mirror for right state
                p0 = (2 * v[4] - 7 * v[3] + 11 * v[2]) / 6.0
                p1 = (-v[3] + 5 * v[2] + 2 * v[1]) / 6.0
                p2 = (2 * v[2] + 5 * v[1] - v[0]) / 6.0
                W_R = omega0 * p0 + omega1 * p1 + omega2 * p2
                # Convert to conservative
                W_tmp_L = W[:, i].copy()
                W_tmp_R = W[:, i].copy()
                W_tmp_L[j] = W_L
                W_tmp_R[j] = W_R
                UL[:, i - 1] = self.equation_system.to_conservative(W_tmp_L)
                UR[:, i - 1] = self.equation_system.to_conservative(W_tmp_R)
        return UL, UR