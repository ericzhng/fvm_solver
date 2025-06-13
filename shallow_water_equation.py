import numpy as np
from .base_equation import EquationSystem


class ShallowWater(EquationSystem):
    """
    1D Shallow Water Equation System.

    Governs the conservation of water height and momentum, including gravitational effects.
    Primitive variables: [height, velocity]
    Conservative variables: [height, momentum]
    """

    def __init__(self, gravity: float = 9.81):
        """
        Initialize the shallow water equation system.

        Args:
            gravity (float): Gravitational acceleration (must be positive).
        """
        if gravity <= 0:
            raise ValueError("gravity must be positive")
        super().__init__(min_value=1e-10)
        self.g = gravity
        self.var_names = ['height', 'velocity']
        self.vel_idx = 1
        self.monitor_idx = 0
        self.num_vars = 2
        self.safety_guard_var_idx = [0]  # only height

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Convert primitive variables to conservative variables.

        Args:
            W (np.ndarray): Primitive variables [height, velocity], shape (2,).

        Returns:
            np.ndarray: Conservative variables [height, momentum], shape (2,).
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        h, u = W
        h = np.maximum(h, self.min_value)
        return np.array([h, h * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables [height, momentum], shape (2,).

        Returns:
            np.ndarray: Primitive variables [height, velocity], shape (2,).
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        h, hu = U
        h = np.maximum(h, self.min_value)
        return np.array([h, hu / h])

    def sound_speed(self, W: np.ndarray) -> float:
        """
        Compute the local wave speed (gravity wave speed).

        Args:
            W (np.ndarray): Primitive variables [height, velocity], shape (2,).

        Returns:
            float: Local wave speed.
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        h = np.maximum(W[0], self.min_value)
        return np.sqrt(self.g * h)

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute the physical flux vector.

        Args:
            U (np.ndarray): Conservative variables [height, momentum], shape (2,).
            W (np.ndarray): Primitive variables [height, velocity], shape (2,).

        Returns:
            np.ndarray: Flux vector, shape (2,).
        """
        if U.shape != (self.num_vars,) or W.shape != (self.num_vars,):
            raise ValueError(f"U and W must have shape ({self.num_vars},)")
        h, u = W
        h = np.maximum(h, self.min_value)
        return np.array([h * u, h * u**2 + 0.5 * self.g * h**2])

    def hllc_numerical_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute the HLLC numerical flux for the shallow water equations.

        Args:
            WL (np.ndarray): Left primitive state [height, velocity], shape (2,).
            WR (np.ndarray): Right primitive state [height, velocity], shape (2,).
            UL (np.ndarray): Left conservative state [height, momentum], shape (2,).
            UR (np.ndarray): Right conservative state [height, momentum], shape (2,).

        Returns:
            np.ndarray: HLLC numerical flux, shape (2,).
        """
        if any(arr.shape != (self.num_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        hL, uL = WL
        hR, uR = WR
        hL = np.maximum(hL, self.min_value)
        hR = np.maximum(hR, self.min_value)

        cL = np.sqrt(self.g * hL)
        cR = np.sqrt(self.g * hR)
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)
        denom = hR * (SR - uR) - hL * (SL - uL)
        S_star = (hR * uR * (SR - uR) - hL * uL * (SL - uL) + 0.5 * self.g * (hR**2 - hL**2)) / (denom + self.min_value)

        hL_star = np.maximum(hL * (SL - uL) / (SL - S_star + self.min_value), self.min_value)
        hR_star = np.maximum(hR * (SR - uR) / (SR - S_star + self.min_value), self.min_value)
        UL_star = np.array([hL_star, hL_star * S_star])
        UR_star = np.array([hR_star, hR_star * S_star])

        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)
        if SL >= 0:
            F = FL
        elif SL <= 0 <= S_star:
            F = FL + SL * (UL_star - UL)
        elif S_star <= 0 <= SR:
            F = FR + SR * (UR_star - UR)
        else:
            F = FR

        return F

    def roe_numerical_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute the Roe numerical flux for the shallow water equations, with entropy fix.

        Args:
            WL (np.ndarray): Left primitive state [height, velocity], shape (2,).
            WR (np.ndarray): Right primitive state [height, velocity], shape (2,).
            UL (np.ndarray): Left conservative state [height, momentum], shape (2,).
            UR (np.ndarray): Right conservative state [height, momentum], shape (2,).

        Returns:
            np.ndarray: Roe numerical flux, shape (2,).
        """
        if any(arr.shape != (self.num_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        hL, uL = WL
        hR, uR = WR
        hL = np.maximum(hL, self.min_value)
        hR = np.maximum(hR, self.min_value)
        sqrt_hL = np.sqrt(hL)
        sqrt_hR = np.sqrt(hR)
        h_roe = sqrt_hL * sqrt_hR
        u_roe = (uL * sqrt_hL + uR * sqrt_hR) / (sqrt_hL + sqrt_hR + self.min_value)
        c_roe = np.sqrt(self.g * h_roe)

        eigenvalues = np.array([u_roe - c_roe, u_roe + c_roe])
        eigenvectors = [np.array([1, u_roe - c_roe]), np.array([1, u_roe + c_roe])]
        delta = 0.1 * c_roe

        delta_U = UR - UL
        alpha_2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe + self.min_value)
        alpha_1 = delta_U[0] - alpha_2
        wave_strengths = np.array([alpha_1, alpha_2])

        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)
        abs_A = np.zeros_like(FL)
        for i in range(2):
            abs_lambda = abs(eigenvalues[i]) if abs(eigenvalues[i]) > delta else (eigenvalues[i]**2 + delta**2) / (2 * delta)
            abs_A += abs_lambda * wave_strengths[i] * eigenvectors[i]
        F = 0.5 * (FL + FR - abs_A)

        return F
