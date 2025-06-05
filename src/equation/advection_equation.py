import numpy as np
from .base_equation import EquationSystem

class AdvectionEquation(EquationSystem):
    """
    1D Linear Advection Equation: u_t + a u_x = 0
    Primitive and conservative variables are the same: [u].
    """
    def __init__(self, advection_speed: float = 1.0):
        super().__init__(min_value=1e-10)
        self.a = advection_speed
        self.vel_idx = 0
        self.monitor_idx = 0
        self.var_names = ['u']
        self.num_vars = 1
        self.safety_guard_var_idx = []

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        return W.copy()

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        return U.copy()

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        # F = a * u
        if U.shape != (self.num_vars,) or W.shape != (self.num_vars,):
            raise ValueError(f"U and W must have shape ({self.num_vars},)")
        return np.array([self.a * W[0]])

    def sound_speed(self, W: np.ndarray) -> float:
        # For linear advection, the "wave speed" is |a|
        return abs(self.a)

    def hllc_numerical_flux(self, WL, WR, UL, UR):
        # For linear advection, HLLC reduces to upwind flux
        a = self.a
        if a >= 0:
            return self.compute_flux(UL, WL)
        else:
            return self.compute_flux(UR, WR)

    def roe_numerical_flux(self, WL, WR, UL, UR):
        # Roe flux is also upwind for linear advection
        a = self.a
        if a >= 0:
            return self.compute_flux(UL, WL)
        else:
            return self.compute_flux(UR, WR)
