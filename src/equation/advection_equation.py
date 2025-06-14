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
        self.var_names = ["u"]
        self.num_vars = len(self.var_names)
        self.vel_idx = 0

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        return W.copy()

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        return U.copy()

    def sound_speed(self, U: np.ndarray) -> float:
        # For linear advection, the "wave speed" is |a|
        return abs(self.a)

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        # F = a * u
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")

        return np.array([self.a * U[0]])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple:
        """Compute the Roe numerical flux for the shallow water equations, with entropy fix.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum, energy]
            U_R (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            np.ndarray: Roe numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        uL = U_L[0]
        uR = U_R[0]

        u_roe = np.sqrt(uL * uR)
        a_roe = self.a

        return a_roe, u_roe

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        # Compute the physical flux.
        a = self.a
        if a >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray):
        # For linear advection, HLLC reduces to upwind flux
        a = self.a
        if a >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray):
        # Roe flux is also upwind for linear advection
        a = self.a
        if a >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray):
        # Roe flux is also upwind for linear advection
        a = self.a
        if a >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)
