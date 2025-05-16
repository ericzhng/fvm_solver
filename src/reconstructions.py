import numpy as np
from .limiters import Limiter

class Reconstruction:
    """Handles state reconstruction for Godunov-type schemes."""
    
    def __init__(self, equation_system, limiter_name: str):
        """Initialize reconstruction with equation system and limiter.
        
        Args:
            equation_system: Equation system instance
            limiter_name: Name of the limiter to use
        """
        self.equation_system = equation_system
        self.limiter = Limiter().get_limiter(limiter_name)
        self.min_var = getattr(equation_system, 'h_min', getattr(equation_system, 'rho_min', 1e-10))

    def piecewise_constant(self, U_ext: np.ndarray, dx: float) -> tuple:
        """Piecewise constant reconstruction (first-order).
        
        Args:
            U_ext: Extended state array (n_vars, n+4)
            dx: Spatial step size
            
        Returns:
            Tuple of left and right states (UL, UR)
        """
        n = U_ext.shape[1]
        UL = U_ext[:, :-1].copy()
        UR = U_ext[:, 1:].copy()
        # Enforce positivity for height/density
        for i in range(n-1):
            UL[0, i] = max(UL[0, i], self.min_var)
            UR[0, i] = max(UR[0, i], self.min_var)
        return UL, UR

    def muscl(self, U_ext: np.ndarray, dx: float) -> tuple:
        """MUSCL reconstruction (second-order) with slope limiting.
        
        Args:
            U_ext: Extended state array (n_vars, n+4)
            dx: Spatial step size
            
        Returns:
            Tuple of left and right states (UL, UR)
        """
        n, n_vars = U_ext.shape[1], U_ext.shape[0]
        UL = np.zeros_like(U_ext[:, :-1])
        UR = np.zeros_like(U_ext[:, :-1])
        slopes = np.zeros_like(U_ext)
        # Compute limited slopes
        for i in range(1, n-1):
            for var in range(n_vars):
                left_slope = (U_ext[var, i] - U_ext[var, i-1]) / dx
                right_slope = (U_ext[var, i+1] - U_ext[var, i]) / dx
                slopes[var, i] = self.limiter(left_slope, right_slope)
        # Reconstruct interface states
        for i in range(n-1):
            for var in range(n_vars):
                UL[var, i] = U_ext[var, i] + 0.5 * dx * slopes[var, i]
                UR[var, i] = U_ext[var, i+1] - 0.5 * dx * slopes[var, i+1]
            UL[0, i] = max(UL[0, i], self.min_var)
            UR[0, i] = max(UR[0, i], self.min_var)
        return UL, UR

    def ppm(self, U_ext: np.ndarray, dx: float) -> tuple:
        """Piecewise Parabolic Method (third-order) reconstruction.
        
        Args:
            U_ext: Extended state array (n_vars, n+4)
            dx: Spatial step size
            
        Returns:
            Tuple of left and right states (UL, UR)
        """
        n, n_vars = U_ext.shape[1], U_ext.shape[0]
        UL = np.zeros_like(U_ext[:, :-1])
        UR = np.zeros_like(U_ext[:, :-1])
        for var in range(n_vars):
            u = U_ext[var, :]
            u_L = np.zeros(n)
            u_R = np.zeros(n)
            # Compute interface values
            for i in range(2, n-2):
                delta_m = self.limiter(u[i] - u[i-1], u[i+1] - u[i])
                u_L[i] = u[i] - 0.5 * delta_m
                u_R[i] = u[i] + 0.5 * delta_m
                # Monotonicity constraints
                if (u_R[i] - u_L[i]) * (u[i] - 0.5 * (u_L[i] + u_R[i])) > (u_R[i] - u_L[i])**2 / 6:
                    u_L[i] = 3 * u[i] - 2 * u_R[i]
                elif (u_R[i] - u_L[i]) * (u[i] - 0.5 * (u_L[i] + u_R[i])) < -(u_R[i] - u_L[i])**2 / 6:
                    u_R[i] = 3 * u[i] - 2 * u_L[i]
            # Assign interface states
            for i in range(1, n-2):
                UL[var, i] = u_R[i]
                UR[var, i] = u_L[i+1]
            # Boundary handling
            UL[var, 0] = u[2]
            UR[var, 0] = u[3]
            UL[var, -2:] = u[-3]
            UR[var, -2:] = u[-2]
            # Enforce positivity
            for i in range(n-1):
                UL[var, i] = max(UL[var, i], self.min_var if var == 0 else UL[var, i])
                UR[var, i] = max(UR[var, i], self.min_var if var == 0 else UR[var, i])
        return UL, UR

    def weno5(self, U_ext: np.ndarray, dx: float) -> tuple:
        """WENO5 reconstruction (fifth-order) for smooth regions.
        
        Args:
            U_ext: Extended state array (n_vars, n+4)
            dx: Spatial step size
            
        Returns:
            Tuple of left and right states (UL, UR)
        """
        n, n_vars = U_ext.shape[1], U_ext.shape[0]
        UL = np.zeros_like(U_ext[:, :-1])
        UR = np.zeros_like(U_ext[:, :-1])
        eps = 1e-10  # Prevent division by zero
        beta_max = 1e10  # Clip smoothness indicators
        for var in range(n_vars):
            u = U_ext[var, :]
            for i in range(2, n-2):
                # Left state (UR[i-1/2])
                v0 = u[i-2:i+1]
                v1 = u[i-1:i+2]
                v2 = u[i:i+3]
                p0 = (2 * v0[0] - 7 * v0[1] + 11 * v0[2]) / 6
                p1 = (-v1[0] + 5 * v1[1] + 2 * v1[2]) / 6
                p2 = (2 * v2[0] + 5 * v2[1] - v2[2]) / 6
                # Smoothness indicators
                beta0 = min(13/12 * (v0[0] - 2 * v0[1] + v0[2])**2 + 1/4 * (v0[0] - 4 * v0[1] + 3 * v0[2])**2, beta_max)
                beta1 = min(13/12 * (v1[0] - 2 * v1[1] + v1[2])**2 + 1/4 * (v1[0] - v1[2])**2, beta_max)
                beta2 = min(13/12 * (v2[0] - 2 * v2[1] + v2[2])**2 + 1/4 * (3 * v2[0] - 4 * v2[1] + v2[2])**2, beta_max)
                alpha0 = 0.1 / (beta0 + eps)**2
                alpha1 = 0.6 / (beta1 + eps)**2
                alpha2 = 0.3 / (beta2 + eps)**2
                w_sum = alpha0 + alpha1 + alpha2
                w0 = alpha0 / (w_sum + 1e-10)
                w1 = alpha1 / (w_sum + 1e-10)
                w2 = alpha2 / (w_sum + 1e-10)
                UL[var, i-1] = w0 * p0 + w1 * p1 + w2 * p2
                # Right state (UL[i+1/2])
                p0 = (-v2[2] + 5 * v2[1] + 2 * v2[0]) / 6
                p1 = (2 * v1[2] + 5 * v1[1] - v1[0]) / 6
                p2 = (11 * v0[2] - 7 * v0[1] + 2 * v0[0]) / 6
                beta0 = min(13/12 * (v2[2] - 2 * v2[1] + v2[0])**2 + 1/4 * (v2[2] - 4 * v2[1] + 3 * v2[0])**2, beta_max)
                beta1 = min(13/12 * (v1[2] - 2 * v1[1] + v1[0])**2 + 1/4 * (v1[2] - v1[0])**2, beta_max)
                beta2 = min(13/12 * (v0[2] - 2 * v0[1] + v0[0])**2 + 1/4 * (3 * v0[2] - 4 * v0[1] + v0[0])**2, beta_max)
                alpha0 = 0.1 / (beta0 + eps)**2
                alpha1 = 0.6 / (beta1 + eps)**2
                alpha2 = 0.3 / (beta2 + eps)**2
                w_sum = alpha0 + alpha1 + alpha2
                w0 = alpha0 / (w_sum + 1e-10)
                w1 = alpha1 / (w_sum + 1e-10)
                w2 = alpha2 / (w_sum + 1e-10)
                UR[var, i-1] = w0 * p0 + w1 * p1 + w2 * p2
            # Boundary handling
            UL[var, 0] = u[2]
            UR[var, 0] = u[3]
            UL[var, -1] = u[-3]
            UR[var, -1] = u[-2]
            # Enforce positivity
            for i in range(n-1):
                UL[var, i] = max(UL[var, i], self.min_var if var == 0 else UL[var, i])
                UR[var, i] = max(UR[var, i], self.min_var if var == 0 else UR[var, i])
        return UL, UR