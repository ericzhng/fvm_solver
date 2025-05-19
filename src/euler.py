import numpy as np
from .equation import EquationSystem


class EulerEquationSystem(EquationSystem):
    """Euler equations for 1D compressible gas dynamics.

    Models conservation of mass, momentum, and energy.
    """

    def __init__(self, gamma: float = 1.4):
        """Initialize the Euler equation system.

        Args:
            gamma (float): Ratio of specific heats (default: 1.4 for air).
        """
        super().__init__()
        self.gamma = gamma
        self.min_var = 1e-10  # Minimum density and pressure
        self.velocity_index = 1
        self.monitored_index = 0
        self.safeguarded_indices : list[int] = [0]
        self.variable_names = ['density', 'velocity', 'pressure']

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Primitive: [density, velocity, pressure]
        Conservative: [density, momentum, total energy]

        Args:
            W (np.ndarray): Primitive variables [density, velocity, pressure].

        Returns:
            np.ndarray: Conservative variables [density, momentum, energy].
        """
        rho, u, p = W
        rho = np.maximum(rho, self.min_var)
        p = np.maximum(p, self.min_var)
        E = p / (self.gamma - 1) + 0.5 * rho * u**2
        return np.array([rho, rho * u, E])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables [density, momentum, energy].

        Returns:
            np.ndarray: Primitive variables [density, velocity, pressure].
        """
        rho, m, E = U
        rho = np.maximum(rho, self.min_var)
        u = m / rho
        p = (E - 0.5 * rho * u**2) * (self.gamma - 1)
        p = np.maximum(p, self.min_var)
        return np.array([rho, u, p])

    def sound_speed(self, W: np.ndarray) -> float:
        """Compute the sound speed for the gas.

        Args:
            W (np.ndarray): Primitive variables [density, velocity, pressure].

        Returns:
            float: Sound speed sqrt(gamma * p / rho).
        """
        rho, p = W[0], W[2]
        rho = np.maximum(rho, self.min_var)
        p = np.maximum(p, self.min_var)
        return np.sqrt(self.gamma * p / rho)

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Flux: [rho*u, rho*u^2 + p, u*(E + p)]

        Args:
            U (np.ndarray): Conservative variables [density, momentum, energy].
            W (np.ndarray): Primitive variables [density, velocity, pressure].

        Returns:
            np.ndarray: Flux vector.
        """
        rho, u, p = W
        E = U[2]  # Total energy
        return np.array([rho * u, rho * u**2 + p, u * (E + p)])

    def hllc_wave_speeds(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute HLLC wave speeds for Euler equations.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity, pressure].
            WR (np.ndarray): Right primitive state [density, velocity, pressure].
            UL (np.ndarray): Left conservative state [density, momentum, energy].
            UR (np.ndarray): Right conservative state [density, momentum, energy].

        Returns:
            tuple: Left, right, and contact wave speeds (S_L, S_R, S_star).
        """
        rhoL, uL, pL = WL
        rhoR, uR, pR = WR
        rhoL = np.maximum(rhoL, self.min_var)
        rhoR = np.maximum(rhoR, self.min_var)
        pL = np.maximum(pL, self.min_var)
        pR = np.maximum(pR, self.min_var)

        cL = np.sqrt(self.gamma * pL / rhoL)
        cR = np.sqrt(self.gamma * pR / rhoR)

        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)

        # Contact wave speed
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = 0.5 * (uL + uR) if abs(denom) < self.min_var else ( 
            pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR) ) / denom

        return S_L, S_R, S_star

    def hllc_intermediate_states(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray,
                                S_L: float, S_R: float, S_star: float) -> tuple:
        """Compute HLLC intermediate states for Euler equations.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity, pressure].
            WR (np.ndarray): Right primitive state [density, velocity, pressure].
            UL (np.ndarray): Left conservative state [density, momentum, energy].
            UR (np.ndarray): Right conservative state [density, momentum, energy].
            S_L (float): Left wave speed.
            S_R (float): Right wave speed.
            S_star (float): Contact wave speed.

        Returns:
            tuple: Left and right intermediate conservative states (UL_star, UR_star).
        """
        rhoL, uL, pL = WL
        rhoR, uR, pR = WR
        rhoL = np.maximum(rhoL, self.min_var)
        rhoR = np.maximum(rhoR, self.min_var)
        pL = np.maximum(pL, self.min_var)
        pR = np.maximum(pR, self.min_var)

        # Intermediate states
        rhoL_star = max(rhoL * (S_L - uL) / (S_L - S_star + self.min_var), self.min_var)
        rhoR_star = max(rhoR * (S_R - uR) / (S_R - S_star + self.min_var), self.min_var)
        EL = UL[2] / (rhoL + self.min_var) + (S_star - uL) * (S_star + pL / (rhoL * (S_L - uL) + self.min_var))
        ER = UR[2] / (rhoR + self.min_var) + (S_star - uR) * (S_star + pR / (rhoR * (S_R - uR) + self.min_var))
        UL_star = np.array([rhoL_star, rhoL_star * S_star, rhoL_star * EL])
        UR_star = np.array([rhoR_star, rhoR_star * S_star, rhoR_star * ER])

        return UL_star, UR_star

    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        rhoL, uL, pL = WL
        rhoR, uR, pR = WR
        hL = (self.to_conservative(WL)[2] + pL) / (rhoL + self.min_var)
        hR = (self.to_conservative(WR)[2] + pR) / (rhoR + self.min_var)
        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + self.min_var)
        h_roe = (hL * np.sqrt(rhoL) + hR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + self.min_var)
        c_roe = np.sqrt((self.gamma - 1) * (h_roe - 0.5 * u_roe**2))
        return np.array([rho_roe, u_roe, h_roe, c_roe])
    
    def roe_eigenstructure(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        _, u_roe, h_roe, c_roe = self.roe_averaged_state(WL, WR)
        eigenvalues = np.array([u_roe - c_roe, u_roe, u_roe + c_roe])
        eigenvectors = [
            np.array([1, u_roe - c_roe, h_roe - u_roe * c_roe]),
            np.array([1, u_roe, 0.5 * u_roe**2]),
            np.array([1, u_roe + c_roe, h_roe + u_roe * c_roe])
        ]
        delta = 0.1 * c_roe
        return eigenvalues, eigenvectors, delta
    
    def roe_wave_strengths(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        _, u_roe, h_roe, c_roe = self.roe_averaged_state(WL, WR)
        delta_U = UR - UL
        delta_rho = delta_U[0]
        delta_rho_u = delta_U[1]
        delta_rho_E = delta_U[2]
        alpha_2 = ((self.gamma - 1) / (c_roe**2 + self.min_var)) * (
            delta_rho * (0.5 * u_roe**2 - h_roe) + delta_rho_u * u_roe + delta_rho_E
        )
        alpha_1 = ((delta_rho - alpha_2) * (u_roe + c_roe) - delta_rho_u) / (2 * c_roe + self.min_var)
        alpha_3 = (delta_rho_u - (delta_rho - alpha_2) * (u_roe - c_roe)) / (2 * c_roe + self.min_var)
        return np.array([alpha_1, alpha_2, alpha_3])
