import numpy as np
from .base_equation import EquationSystem


class IsentropicGas(EquationSystem):
    """Isentropic gas equation system for 1D flows.

    Models conservation of mass and momentum under isentropic conditions.
    """

    def __init__(self, gamma: float = 1.4, k: float = 1.0):
        """Initialize the isentropic gas system.

        Args:
            gamma (float): Ratio of specific heats (defaU_Lt: 1.4).
            k (float): Gas constant (defaU_Lt: 1.0).
        """
        super().__init__(min_value=1e-10)
        self.gamma = gamma
        self.k = k
        self.var_names = ["density", "velocity"]
        self.num_vars = len(self.var_names)
        self.vel_idx = 1

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Primitive: [density, velocity]
        Conservative: [density, momentum = density * velocity]

        Args:
            W (np.ndarray): Primitive variables [density, velocity].

        RetU_Rns:
            np.ndarray: Conservative variables [density, momentum].
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        rho, u = W
        rho = np.maximum(rho, self.min_value)
        return np.array([rho, rho * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables [density, momentum].

        RetU_Rns:
            np.ndarray: Primitive variables [density, velocity].
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, m = U
        rho = np.maximum(rho, self.min_value)
        return np.array([rho, m / rho])

    def sound_speed(self, U: np.ndarray) -> float:
        """Compute the sound speed for the isentropic gas.

        Args:
            W (np.ndarray): Primitive variables [density, velocity].

        RetU_Rns:
            float: Sound speed sqrt(gamma * k * rho^(gamma-1)).
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho = U[0]
        rho = np.maximum(rho, self.min_value)
        return np.sqrt(self.gamma * self.k * rho ** (self.gamma - 1))

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Flux: [rho*u, rho*u^2 + k*rho^gamma]

        Args:
            U (np.ndarray): Conservative variables [density, momentum].
            W (np.ndarray): Primitive variables [density, velocity].

        RetU_Rns:
            np.ndarray: Flux vector.
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, m = U
        rho = np.maximum(rho, self.min_value)
        u = m / rho
        p = self.k * rho**self.gamma
        return np.array([rho * u, rho * u**2 + p])

    # ---------------------------------------------------- #
    # flux methods that has to be defined per equation wise
    # ---------------------------------------------------- #

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute HLLC wave speeds, intermediate states, and flux for isentropic gas.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum].
            U_R (np.ndarray): Right conservative state [density, momentum].

        RetU_Rns:
            np.ndarray: HLLC numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)

        # Extract variables
        rhoL, uL = W_L
        rhoR, uR = W_R
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)

        # Compute wave speeds
        cL = self.sound_speed(W_L)
        cR = self.sound_speed(W_R)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = (
            0.5 * (uL + uR)
            if abs(denom) < self.min_value
            else (pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / denom
        )

        # Compute intermediate states
        rho_star_L = max(
            rhoL * (S_L - uL) / (S_L - S_star + self.min_value), self.min_value
        )
        rho_star_R = max(
            rhoR * (S_R - uR) / (S_R - S_star + self.min_value), self.min_value
        )
        U_L_star = np.array([rho_star_L, rho_star_L * S_star])
        U_R_star = np.array([rho_star_R, rho_star_R * S_star])

        # Compute flux
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if S_L >= 0:
            F = FL
        elif S_L <= 0 <= S_star:
            F = FL + S_L * (U_L_star - U_L)
        elif S_star <= 0 <= S_R:
            F = FR + S_R * (U_R_star - U_R)
        else:
            F = FR

        return F

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute Roe-averaged state, eigenstructU_Re, wave strengths, and flux for isentropic gas.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum].
            U_R (np.ndarray): Right conservative state [density, momentum].

        RetU_Rns:
            np.ndarray: Roe numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)

        rhoL, uL = W_L
        rhoR, uR = W_R
        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (
            np.sqrt(rhoL) + np.sqrt(rhoR) + self.min_value
        )
        c_roe = self.sound_speed(np.array([rho_roe, u_roe]))

        eigenvalues = np.array([u_roe - c_roe, u_roe + c_roe])
        eigenvectors = [np.array([1, u_roe - c_roe]), np.array([1, u_roe + c_roe])]
        delta = 0.1 * c_roe

        # EigenstructU_Re and wave strengths
        delta_U = U_R - U_L
        alpha_2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (
            -2 * c_roe + self.min_value
        )
        alpha_1 = delta_U[0] - alpha_2
        wave_strengths = np.array([alpha_1, alpha_2])

        # Compute flux
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        abs_A = np.zeros_like(FL)
        for i in range(len(eigenvalues)):
            abs_lambda = (
                abs(eigenvalues[i])
                if abs(eigenvalues[i]) > delta
                else (eigenvalues[i] + np.sqrt(eigenvalues[i] ** 2 + delta**2)) / 2.0
            )
            abs_A += abs_lambda * wave_strengths[i] * eigenvectors[i]
        F = 0.5 * (FL + FR - abs_A)

        return F
