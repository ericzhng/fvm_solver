import numpy as np
from .equation import EquationSystem


class IsentropicGasSystem(EquationSystem):
    """Isentropic gas equation system for 1D flows.

    Models conservation of mass and momentum under isentropic conditions.
    """

    def __init__(self, gamma: float = 1.4, k: float = 1.0):
        """Initialize the isentropic gas system.

        Args:
            gamma (float): Ratio of specific heats (default: 1.4).
            k (float): Gas constant (default: 1.0).
        """
        super().__init__()
        self.gamma = gamma
        self.k = k
        self.min_var = 1e-10  # Minimum density
        self.velocity_index = 1
        self.monitored_index = 0
        self.safeguarded_indices : list[int] = [0]
        self.variable_names = ['density', 'velocity']

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Primitive: [density, velocity]
        Conservative: [density, momentum = density * velocity]

        Args:
            W (np.ndarray): Primitive variables [density, velocity].

        Returns:
            np.ndarray: Conservative variables [density, momentum].
        """
        rho, u = W
        rho = np.maximum(rho, self.min_var)
        return np.array([rho, rho * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables [density, momentum].

        Returns:
            np.ndarray: Primitive variables [density, velocity].
        """
        rho, m = U
        rho = np.maximum(rho, self.min_var)
        return np.array([rho, m / rho])

    def sound_speed(self, W: np.ndarray) -> float:
        """Compute the sound speed for the isentropic gas.

        Args:
            W (np.ndarray): Primitive variables [density, velocity].

        Returns:
            float: Sound speed sqrt(gamma * k * rho^(gamma-1)).
        """
        rho = np.maximum(W[0], self.min_var)
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
        rho, u = W
        rho = np.maximum(rho, self.min_var)
        p = self.k * rho**self.gamma
        return np.array([rho * u, rho * u**2 + p])

    def hllc_wave_speeds(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute HLLC wave speeds for isentropic gas.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity].
            WR (np.ndarray): Right primitive state [density, velocity].
            UL (np.ndarray): Left conservative state [density, momentum].
            UR (np.ndarray): Right conservative state [density, momentum].

        Returns:
            tuple: Left, right, and contact wave speeds (S_L, S_R, S_star).
        """
        rhoL, uL = WL
        rhoR, uR = WR
        rhoL = np.maximum(rhoL, self.min_var)
        rhoR = np.maximum(rhoR, self.min_var)
        cL = self.sound_speed(WL)
        cR = self.sound_speed(WR)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        # Contact wave speed: weighted average
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = 0.5 * (uL + uR) if abs(denom) < 1e-10 else (
            pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)
        ) / denom
        return S_L, S_R, S_star

    def hllc_intermediate_states(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray,
                                S_L: float, S_R: float, S_star: float) -> tuple:
        """Compute HLLC intermediate states for isentropic gas.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity].
            WR (np.ndarray): Right primitive state [density, velocity].
            UL (np.ndarray): Left conservative state [density, momentum].
            UR (np.ndarray): Right conservative state [density, momentum].
            S_L (float): Left wave speed.
            S_R (float): Right wave speed.
            S_star (float): Contact wave speed.

        Returns:
            tuple: Left and right intermediate conservative states (UL_star, UR_star).
        """
        rhoL, uL = WL
        rhoR, uR = WR
        rhoL = np.maximum(rhoL, self.min_var)
        rhoR = np.maximum(rhoR, self.min_var)
        # Intermediate density
        rho_star_L = rhoL * (S_L - uL) / (S_L - S_star + 1e-10)
        rho_star_R = rhoR * (S_R - uR) / (S_R - S_star + 1e-10)
        # Intermediate states: [rho, rho*u]
        UL_star = np.array([rho_star_L, rho_star_L * S_star])
        UR_star = np.array([rho_star_R, rho_star_R * S_star])
        return UL_star, UR_star

    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        rhoL, uL = WL
        rhoR, uR = WR
        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + 1e-10)
        c_roe = self.sound_speed(np.array([rho_roe, u_roe]))
        return np.array([rho_roe, u_roe, c_roe])
    
    def roe_eigenstructure(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute Roe eigenstructure for isentropic gas.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity].
            WR (np.ndarray): Right primitive state [density, velocity].
            UL (np.ndarray): Left conservative state [density, momentum].
            UR (np.ndarray): Right conservative state [density, momentum].

        Returns:
            tuple: Eigenvalues, eigenvectors, and entropy fix parameter (delta).
        """
        rho_roe, u_roe, c_roe = self.roe_averaged_state(WL, WR)
        eigenvalues = np.array([u_roe - c_roe, u_roe + c_roe])
        eigenvectors = [
            np.array([1, u_roe - c_roe]),
            np.array([1, u_roe + c_roe])
        ]
        delta = 0.1 * c_roe
        return eigenvalues, eigenvectors, delta
    
    def roe_wave_strengths(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute Roe wave strengths for isentropic gas.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity].
            WR (np.ndarray): Right primitive state [density, velocity].
            UL (np.ndarray): Left conservative state [density, momentum].
            UR (np.ndarray): Right conservative state [density, momentum].

        Returns:
            np.ndarray: Wave strength coefficients (alpha).
        """
        rho_roe, u_roe, c_roe = self.roe_averaged_state(WL, WR)
        delta_U = UR - UL
        alpha_2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe + 1e-10)
        alpha_1 = delta_U[0] - alpha_2
        return np.array([alpha_1, alpha_2])
