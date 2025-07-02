import numpy as np
from .equation_base import EqnBase


class EqnTrafficLWR(EqnBase):
    """
    1D Linear Advection Equation: u_t + a u_x = 0
    Primitive and conservative variables are the same: [u].
    """

    def __init__(self, rhoMax: float = 1.0, vMax: float = 1.0):
        super().__init__(min_value=1e-10)
        self.rhoMax = rhoMax
        self.vMax = vMax
        self.var_names = ["density"]
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
        rho = U[0]
        return np.array([rho * self.vMax * (1 - rho / self.rhoMax)])

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
        rhoL = U_L[0]
        rhoR = U_R[0]

        rho_roe = (rhoL + rhoR) / 2.0

        return rho_roe, 0

    # ---------------------------------------------------- #
    # flux methods that has to be defined per equation wise
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray):
        # Roe flux is also upwind for linear advection

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # left state
        rhoL = U_L[0]
        rhoR = U_R[0]
        rho_roe = (rhoL + rhoR) / 2.0

        # Differences in primitive variables
        du = rhoR - rhoL

        # eigenvalues and eigenvectors
        lambda1 = self.vMax * (1 - 2 * rho_roe / self.rhoMax)

        # wave strength
        alpha = du

        # Add the matrix dissipation term to complete the Roe flux
        Roe = (FL + FR - (lambda1 * alpha)) / 2

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
