import numpy as np
from .equation_base import EqnBase


class EqnKK(EqnBase):
    """
    1D Linear Advection Equation: u_t + a u_x = 0
    Primitive and conservative variables are the same: [u].
    """

    def __init__(self):
        super().__init__(min_value=1e-10)
        self.var_names = ["u", "v"]
        self.num_vars = len(self.var_names)
        self.vel_idx = None

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
        return 0

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        # F = a * u
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        r = np.sqrt(U[0] * U[0] + U[1] * U[1])

        return np.array([U[0] * r, U[1] * r])

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
        uL, vL = U_L
        uR, vR = U_R
        rL = np.sqrt(U_L[0] * U_L[0] + U_L[1] * U_L[1])
        rR = np.sqrt(U_R[0] * U_R[0] + U_R[1] * U_R[1])

        # Roe averages
        sqrt_rL = np.sqrt(rL)
        sqrt_rR = np.sqrt(rR)
        u_roe = (uL * sqrt_rL + uR * sqrt_rR) / np.maximum(
            sqrt_rL + sqrt_rR, self.min_value
        )
        v_roe = (vL * sqrt_rL + vR * sqrt_rR) / np.maximum(
            sqrt_rL + sqrt_rR, self.min_value
        )
        r_roe = np.sqrt(u_roe * u_roe + v_roe * v_roe)

        return r_roe, 0

    # ---------------------------------------------------- #
    # flux methods that has to be defined per equation wise
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray):
        # Roe flux is also upwind for linear advection

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # left state
        uL, vL = U_L
        uR, vR = U_R
        rL = np.sqrt(U_L[0] * U_L[0] + U_L[1] * U_L[1])
        rR = np.sqrt(U_R[0] * U_R[0] + U_R[1] * U_R[1])

        # Roe averages
        sqrt_rL = np.sqrt(rL)
        sqrt_rR = np.sqrt(rR)
        u_roe = (uL * sqrt_rL + uR * sqrt_rR) / np.maximum(
            sqrt_rL + sqrt_rR, self.min_value
        )
        v_roe = (vL * sqrt_rL + vR * sqrt_rR) / np.maximum(
            sqrt_rL + sqrt_rR, self.min_value
        )
        r_roe = np.sqrt(u_roe * u_roe + v_roe * v_roe)

        # Differences in primitive variables
        du = uR - uL
        dv = vR - vL

        # eigenvalues and eigenvectors
        lambda2 = 2 * r_roe

        # wave strength
        alpha = (u_roe * du + v_roe * dv) / r_roe**2
        vec2 = np.array([du, dv])

        # Add the matrix dissipation term to complete the Roe flux
        Roe = (FL + FR - (lambda2 * alpha * vec2)) / 2

        return Roe

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        # Compute the physical flux.

        # left & right states
        uL = U_L[0]
        uR = U_R[0]
        u_roe = (uL + uR) / 2.0

        if u_roe >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray):
        # For linear advection, HLLC reduces to upwind flux

        # left & right states
        uL = U_L[0]
        uR = U_R[0]
        u_roe = (uL + uR) / 2.0

        if u_roe >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray):
        # Roe flux is also upwind for linear advection

        # left & right states
        uL = U_L[0]
        uR = U_R[0]
        u_roe = (uL + uR) / 2.0

        if u_roe >= 0:
            return self.compute_flux(U_L)
        else:
            return self.compute_flux(U_R)
