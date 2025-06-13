import numpy as np
from .base_equation import EquationSystem


class IsentropicGas(EquationSystem):
    """Isentropic gas equation system for 1D flows.

    Models conservation of mass and momentum under isentropic conditions.
    """

    def __init__(self, gamma: float = 1.4, k: float = 1.0):
        """Initialize the isentropic gas system.

        Args:
            gamma (float): Ratio of specific heats (default: 1.4).
            k (float): Gas constant (default: 1.0).
        """
        super().__init__(min_value=1e-10)
        self.gamma = gamma
        self.k = k
        self.vel_idx = 1
        self.monitored_index = 0
        self.safeguarded_indices : list[int] = [0]
        self.var_names = ['density', 'velocity']
        self.num_vars = 2
        self.safety_guard_var_idx = [0]  # only density

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Primitive: [density, velocity]
        Conservative: [density, momentum = density * velocity]

        Args:
            W (np.ndarray): Primitive variables [density, velocity].

        Returns:
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

        Returns:
            np.ndarray: Primitive variables [density, velocity].
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, m = U
        rho = np.maximum(rho, self.min_value)
        return np.array([rho, m / rho])

    def sound_speed(self, W: np.ndarray) -> float:
        """Compute the sound speed for the isentropic gas.

        Args:
            W (np.ndarray): Primitive variables [density, velocity].

        Returns:
            float: Sound speed sqrt(gamma * k * rho^(gamma-1)).
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        rho = np.maximum(W[0], self.min_value)
        return np.sqrt(self.gamma * self.k * rho**(self.gamma - 1))

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Flux: [rho*u, rho*u^2 + k*rho^gamma]

        Args:
            U (np.ndarray): Conservative variables [density, momentum].
            W (np.ndarray): Primitive variables [density, velocity].

        Returns:
            np.ndarray: Flux vector.
        """
        if U.shape != (self.num_vars,) or W.shape != (self.num_vars,):
            raise ValueError(f"U and W must have shape ({self.num_vars},)")
        rho, u = W
        rho = np.maximum(rho, self.min_value)
        p = self.k * rho**self.gamma
        return np.array([rho * u, rho * u**2 + p])

    def hllc_numerical_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute HLLC wave speeds, intermediate states, and flux for isentropic gas.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity].
            WR (np.ndarray): Right primitive state [density, velocity].
            UL (np.ndarray): Left conservative state [density, momentum].
            UR (np.ndarray): Right conservative state [density, momentum].

        Returns:
            np.ndarray: HLLC numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # Extract variables
        rhoL, uL = WL
        rhoR, uR = WR
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)

        # Compute wave speeds
        cL = self.sound_speed(WL)
        cR = self.sound_speed(WR)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = 0.5 * (uL + uR) if abs(denom) < self.min_value else (
            pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / denom

        # Compute intermediate states
        rho_star_L = max(rhoL * (S_L - uL) / (S_L - S_star + self.min_value), self.min_value)
        rho_star_R = max(rhoR * (S_R - uR) / (S_R - S_star + self.min_value), self.min_value)
        UL_star = np.array([rho_star_L, rho_star_L * S_star])
        UR_star = np.array([rho_star_R, rho_star_R * S_star])

        # Compute flux
        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)
        if S_L >= 0:
            F = FL
        elif S_L <= 0 <= S_star:
            F = FL + S_L * (UL_star - UL)
        elif S_star <= 0 <= S_R:
            F = FR + S_R * (UR_star - UR)
        else:
            F = FR

        return F

    def roe_numerical_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute Roe-averaged state, eigenstructure, wave strengths, and flux for isentropic gas.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity].
            WR (np.ndarray): Right primitive state [density, velocity].
            UL (np.ndarray): Left conservative state [density, momentum].
            UR (np.ndarray): Right conservative state [density, momentum].

        Returns:
            np.ndarray: Roe numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")
        
        rhoL, uL = WL
        rhoR, uR = WR
        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + self.min_value)
        c_roe = self.sound_speed(np.array([rho_roe, u_roe]))

        eigenvalues = np.array([u_roe - c_roe, u_roe + c_roe])
        eigenvectors = [
            np.array([1, u_roe - c_roe]),
            np.array([1, u_roe + c_roe])
        ]
        delta = 0.1 * c_roe

        # Eigenstructure and wave strengths
        delta_U = UR - UL
        alpha_2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe + self.min_value)
        alpha_1 = delta_U[0] - alpha_2
        wave_strengths = np.array([alpha_1, alpha_2])

        # Compute flux
        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)
        abs_A = np.zeros_like(FL)
        for i in range(len(eigenvalues)):
            abs_lambda = abs(eigenvalues[i]) if abs(eigenvalues[i]) > delta else (eigenvalues[i] + np.sqrt(eigenvalues[i]**2 + delta**2)) / 2.0
            abs_A += abs_lambda * wave_strengths[i] * eigenvectors[i]
        F = 0.5 * (FL + FR - abs_A)

        return F
