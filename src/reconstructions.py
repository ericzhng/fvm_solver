import numpy as np
from .equation import EquationSystem
from .limiters import Limiter

class Reconstruction:
    """Handles state reconstruction for Godunov-type schemes."""
    
    def __init__(self, equation_system, limiter_name: str, limiter: str = 'minmod'):
        """Initialize reconstruction with equation system and limiter.
        
        Args:
            equation_system: Equation system instance
            limiter_name: Name of the limiter to use
        """
        self.equation_system = equation_system
        self.min_var = equation_system.min_var
        self.safeguarded_indices = equation_system.safeguarded_indices
        self.limiter = Limiter(limiter)
    
    def _apply_safeguards(self, W: np.ndarray) -> np.ndarray:
        """Apply safeguards to primitive variables."""
        for idx in self.safeguarded_indices:
            W[idx] = np.maximum(W[idx], self.min_var)
        return W
    
    def piecewise_constant(self, U: np.ndarray, dx: float) -> tuple:
        """Piecewise constant reconstruction (first-order).
        
        Args:
            U: Extended state array (n_vars, n+4)
            dx: Spatial step size
            
        Returns:
            Tuple of left and right states (UL, UR)
        """
        n = U.shape[1]
        UL = np.zeros_like(U[:, :-1])
        UR = np.zeros_like(U[:, :-1])
        for i in range(n - 1):
            UL[:, i] = U[:, i]
            UR[:, i] = U[:, i + 1]
        return UL, UR
    
    def muscl(self, U: np.ndarray, dx: float) -> tuple:
        """MUSCL reconstruction (second-order) with slope limiting.
        
        Args:
            U: Extended state array (n_vars, n+4)
            dx: Spatial step size
            
        Returns:
            Tuple of left and right states (UL, UR)
        """
        n = U.shape[1]
        UL = np.zeros_like(U[:, :-1])
        UR = np.zeros_like(U[:, :-1])
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n)]).T
        
        for i in range(2, n - 2):
            for j in range(U.shape[0]):
                sigma_L = self.limiter.limit(
                    (W[j, i] - W[j, i - 1]) / dx,
                    (W[j, i - 1] - W[j, i - 2]) / dx
                )
                sigma_R = self.limiter.limit(
                    (W[j, i + 1] - W[j, i]) / dx,
                    (W[j, i + 2] - W[j, i + 1]) / dx
                )
                W_L = W[j, i] - 0.5 * dx * sigma_L
                W_R = W[j, i] + 0.5 * dx * sigma_R
                UL[j, i - 1] = self.equation_system.to_conservative(
                    self._apply_safeguards(np.array([W_L if k == j else W[k, i] for k in range(U.shape[0])]))
                )[j]
                UR[j, i - 1] = self.equation_system.to_conservative(
                    self._apply_safeguards(np.array([W_R if k == j else W[k, i] for k in range(U.shape[0])]))
                )[j]
        
        return UL, UR
    
    def ppm(self, U: np.ndarray, dx: float) -> tuple:
        """Piecewise Parabolic Method (third-order) reconstruction.
        
        Args:
            U: Extended state array (n_vars, n+4)
            dx: Spatial step size
            
        Returns:
            Tuple of left and right states (UL, UR)
        """
        n = U.shape[1]
        UL = np.zeros_like(U[:, :-1])
        UR = np.zeros_like(U[:, :-1])
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n)]).T
        
        for i in range(2, n - 2):
            for j in range(U.shape[0]):
                W_avg = 0.5 * (W[j, i + 1] + W[j, i])
                delta_W = W[j, i + 1] - W[j, i]
                W_6 = 6 * (W_avg - 0.5 * (W[j, i] + W[j, i + 1]))
                
                W_L = W[j, i] + 0.5 * delta_W + W_6 / 6.0
                W_R = W[j, i + 1] - 0.5 * delta_W + W_6 / 6.0
                
                if (W_R - W_L) * (W_avg - W_L) <= 0:
                    W_L = W_avg
                if (W_R - W_L) * (W_R - W_avg) <= 0:
                    W_R = W_avg
                
                W_L = max(min(W[j, i], W[j, i + 1]), min(W_L, max(W[j, i], W[j, i + 1])))
                W_R = max(min(W[j, i], W[j, i + 1]), min(W_R, max(W[j, i], W[j, i + 1])))
                
                UL[j, i - 1] = self.equation_system.to_conservative(
                    self._apply_safeguards(np.array([W_L if k == j else W[k, i] for k in range(U.shape[0])]))
                )[j]
                UR[j, i - 1] = self.equation_system.to_conservative(
                    self._apply_safeguards(np.array([W_R if k == j else W[k, i] for k in range(U.shape[0])]))
                )[j]
        
        return UL, UR
    
    def weno5(self, U: np.ndarray, dx: float) -> tuple:
        """WENO5 reconstruction (fifth-order) for smooth regions.
        
        Args:
            U: Extended state array (n_vars, n+4)
            dx: Spatial step size
            
        Returns:
            Tuple of left and right states (UL, UR)
        """
        n = U.shape[1]
        UL = np.zeros_like(U[:, :-1])
        UR = np.zeros_like(U[:, :-1])
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n)]).T
        # Dynamic epsilon based on state magnitude and min_var
        epsilon = max(1e-6, self.min_var * np.max(np.abs(W)))
        
        for i in range(2, n - 2):
            for j in range(U.shape[0]):
                v = W[j, i - 2:i + 3]
                
                beta0 = 13.0 / 12.0 * (v[0] - 2 * v[1] + v[2])**2 + 0.25 * (v[0] - 4 * v[1] + 3 * v[2])**2
                beta1 = 13.0 / 12.0 * (v[1] - 2 * v[2] + v[3])**2 + 0.25 * (v[1] - v[3])**2
                beta2 = 13.0 / 12.0 * (v[2] - 2 * v[3] + v[4])**2 + 0.25 * (3 * v[2] - 4 * v[3] + v[4])**2
                
                d0, d1, d2 = 0.1, 0.6, 0.3
                alpha0 = d0 / (beta0 + epsilon)**2
                alpha1 = d1 / (beta1 + epsilon)**2
                alpha2 = d2 / (beta2 + epsilon)**2
                sum_alpha = alpha0 + alpha1 + alpha2
                
                omega0 = alpha0 / sum_alpha
                omega1 = alpha1 / sum_alpha
                omega2 = alpha2 / sum_alpha
                
                p0 = (2 * v[0] - 7 * v[1] + 11 * v[2]) / 6.0
                p1 = (-v[1] + 5 * v[2] + 2 * v[3]) / 6.0
                p2 = (2 * v[2] + 5 * v[3] - v[4]) / 6.0
                W_R = omega0 * p0 + omega1 * p1 + omega2 * p2
                
                beta0 = 13.0 / 12.0 * (v[2] - 2 * v[1] + v[0])**2 + 0.25 * (3 * v[2] - 4 * v[1] + v[0])**2
                beta1 = 13.0 / 12.0 * (v[3] - 2 * v[2] + v[1])**2 + 0.25 * (v[3] - v[1])**2
                beta2 = 13.0 / 12.0 * (v[4] - 2 * v[3] + v[2])**2 + 0.25 * (v[4] - 4 * v[3] + 3 * v[2])**2
                
                alpha0 = d0 / (beta0 + epsilon)**2
                alpha1 = d1 / (beta1 + epsilon)**2
                alpha2 = d2 / (beta2 + epsilon)**2
                sum_alpha = alpha0 + alpha1 + alpha2
                
                omega0 = alpha0 / sum_alpha
                omega1 = alpha1 / sum_alpha
                omega2 = alpha2 / sum_alpha
                
                p0 = (2 * v[2] + 5 * v[1] - v[0]) / 6.0
                p1 = (-v[3] + 5 * v[2] + 2 * v[1]) / 6.0
                p2 = (11 * v[2] - 7 * v[3] + 2 * v[4]) / 6.0
                W_L = omega0 * p0 + omega1 * p1 + omega2 * p2
                
                UL[j, i - 1] = self.equation_system.to_conservative(
                    self._apply_safeguards(np.array([W_L if k == j else W[k, i] for k in range(U.shape[0])]))
                )[j]
                UR[j, i - 1] = self.equation_system.to_conservative(
                    self._apply_safeguards(np.array([W_R if k == j else W[k, i] for k in range(U.shape[0])]))
                )[j]
        
        return UL, UR
