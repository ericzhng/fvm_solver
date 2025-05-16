import numpy as np

class EquationSystem:
    """Base class for hyperbolic conservation law systems."""
    
    def __init__(self, min_var: float = 1e-10):
        """Initialize equation system with minimum variable threshold.
        
        Args:
            min_var: Minimum threshold for safeguarded variables
        """
        self.min_var = min_var
        self.velocity_index = None  # Index of velocity in primitive variables
        self.monitored_index = None  # Index of monitored primitive variable
        self.safeguarded_indices = []  # Indices of conservative variables to safeguard
    
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative."""
        raise NotImplementedError
    
    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive."""
        raise NotImplementedError
    
    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute physical flux for given state."""
        raise NotImplementedError
    
    def sound_speed(self, W: np.ndarray) -> float:
        """Compute sound speed for given primitive state."""
        raise NotImplementedError
    
    def get_variable_names(self) -> list:
        """Get names of primitive variables."""
        raise NotImplementedError

    def hllc_wave_speeds(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute HLLC wave speeds (S_L, S_R, S^*).
        
        Returns:
            Tuple (S_L, S_R, S^*)
        """
        raise NotImplementedError
    
    def hllc_intermediate_states(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray, S_L: float, S_R: float, S_star: float) -> tuple:
        """Compute HLLC intermediate states (U_L^*, U_R^*).
        
        Returns:
            Tuple (U_L^*, U_R^*)
        """
        raise NotImplementedError

    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Compute Roe-averaged state variables numerically (default)."""
        UL = self.to_conservative(WL)
        UR = self.to_conservative(WR)
        U_roe = 0.5 * (UL + UR)
        W_roe = self.to_primitive(U_roe)
        c_roe = self.sound_speed(W_roe)
        # Default: velocity and sound speed
        return np.array([W_roe[self.velocity_index], c_roe])
    
    def roe_eigenstructure(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute Roe eigenvalues, eigenvectors, and entropy fix parameter.
        
        Returns:
            Tuple (eigenvalues, eigenvectors, delta)
        """
        raise NotImplementedError
    
    def roe_wave_strengths(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute Roe wave strengths."""
        raise NotImplementedError


class ShallowWaterSystem(EquationSystem):
    """Shallow water equations system."""
    
    def __init__(self, g: float = 9.81, h_min: float = 1e-10):
        """Initialize shallow water system."""
        super().__init__(h_min)
        self.g = g
        self.velocity_index = 1
        self.monitored_index = 0
        self.safeguarded_indices = [0]  # Safeguard height (h)
    
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert [h, u] to [h, hu]."""
        h, u = W
        return np.array([h, h * u])
    
    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert [h, hu] to [h, u]."""
        h, hu = U
        u = hu / (h + self.min_var)
        return np.array([h, u])
    
    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute flux [hu, hu^2 + 0.5 g h^2]."""
        h, u = W
        return np.array([h * u, h * u**2 + 0.5 * self.g * h**2])
    
    def sound_speed(self, W: np.ndarray) -> float:
        """Compute sound speed sqrt(g h)."""
        h = W[0]
        return np.sqrt(self.g * h)
    
    def get_variable_names(self) -> list:
        """Return variable names."""
        return ['height', 'velocity']

    def hllc_wave_speeds(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        hL, uL = WL
        hR, uR = WR
        cL = self.sound_speed(WL)
        cR = self.sound_speed(WR)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        denom = hR * (S_R - uR) - hL * (S_L - uL)
        S_star = 0.5 * (uL + uR) if abs(denom) < 1e-10 else (
            hR * uR * (S_R - uR) - hL * uL * (S_L - uL) + 0.5 * self.g * (hR**2 - hL**2)
        ) / denom
        return S_L, S_R, S_star
    
    def hllc_intermediate_states(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray, S_L: float, S_R: float, S_star: float) -> tuple:
        hL, uL = WL
        hR, uR = WR
        hL_star = max(hL * (S_L - uL) / (S_L - S_star + 1e-10), self.min_var)
        hR_star = max(hR * (S_R - uR) / (S_R - S_star + 1e-10), self.min_var)
        UL_star = np.array([hL_star, hL_star * S_star])
        UR_star = np.array([hR_star, hR_star * S_star])
        return UL_star, UR_star

    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        hL, uL = WL
        hR, uR = WR
        h_roe = np.sqrt(hL * hR)
        u_roe = (uL * np.sqrt(hL) + uR * np.sqrt(hR)) / (np.sqrt(hL) + np.sqrt(hR) + 1e-10)
        c_roe = np.sqrt(self.g * h_roe)
        return np.array([h_roe, u_roe, c_roe])
    
    def roe_eigenstructure(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        h_roe, u_roe, c_roe = self.roe_averaged_state(WL, WR)
        eigenvalues = np.array([u_roe - c_roe, u_roe + c_roe])
        eigenvectors = [
            np.array([1, u_roe - c_roe]),
            np.array([1, u_roe + c_roe])
        ]
        delta = 0.1 * c_roe
        return eigenvalues, eigenvectors, delta
    
    def roe_wave_strengths(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        h_roe, u_roe, c_roe = self.roe_averaged_state(WL, WR)
        delta_U = UR - UL
        alpha_2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe + 1e-10)
        alpha_1 = delta_U[0] - alpha_2
        return np.array([alpha_1, alpha_2])


class EulerEquationSystem(EquationSystem):
    """Euler equations system for compressible gas dynamics."""
    
    def __init__(self, gamma: float = 1.4, rho_min: float = 1e-10, p_min: float = 1e-10):
        """Initialize Euler system."""
        super().__init__(rho_min)
        self.gamma = gamma
        self.p_min = p_min
        self.velocity_index = 1
        self.monitored_index = 0
        self.safeguarded_indices = [0]  # Safeguard density (rho)
    
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert [rho, u, p] to [rho, rho u, E]."""
        rho, u, p = W
        E = p / (self.gamma - 1) + 0.5 * rho * u**2
        return np.array([rho, rho * u, E])
    
    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert [rho, rho u, E] to [rho, u, p]."""
        rho, rho_u, E = U
        u = rho_u / (rho + self.min_var)
        p = (self.gamma - 1) * (E - 0.5 * rho * u**2)
        p = max(p, self.p_min)
        return np.array([rho, u, p])
    
    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute flux [rho u, rho u^2 + p, u (E + p)]."""
        rho, u, p = W
        E = U[2]
        return np.array([rho * u, rho * u**2 + p, u * (E + p)])
    
    def sound_speed(self, W: np.ndarray) -> float:
        """Compute sound speed sqrt(gamma p / rho)."""
        rho, p = W[0], W[2]
        return np.sqrt(self.gamma * p / (rho + self.min_var))
    
    def get_variable_names(self) -> list:
        """Return variable names."""
        return ['density', 'velocity', 'pressure']

    def hllc_wave_speeds(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        rhoL, uL, pL = WL
        rhoR, uR, pR = WR
        cL = self.sound_speed(WL)
        cR = self.sound_speed(WR)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = 0.5 * (uL + uR) if abs(denom) < 1e-10 else (
            pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)
        ) / denom
        return S_L, S_R, S_star
    
    def hllc_intermediate_states(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray, S_L: float, S_R: float, S_star: float) -> tuple:
        rhoL, uL, pL = WL
        rhoR, uR, pR = WR
        rhoL_star = max(rhoL * (S_L - uL) / (S_L - S_star + 1e-10), self.min_var)
        rhoR_star = max(rhoR * (S_R - uR) / (S_R - S_star + 1e-10), self.min_var)
        EL = UL[2] / (rhoL + 1e-10) + (S_star - uL) * (S_star + pL / (rhoL * (S_L - uL) + 1e-10))
        ER = UR[2] / (rhoR + 1e-10) + (S_star - uR) * (S_star + pR / (rhoR * (S_R - uR) + 1e-10))
        UL_star = np.array([rhoL_star, rhoL_star * S_star, rhoL_star * EL])
        UR_star = np.array([rhoR_star, rhoR_star * S_star, rhoR_star * ER])
        return UL_star, UR_star

    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        rhoL, uL, pL = WL
        rhoR, uR, pR = WR
        hL = (self.to_conservative(WL)[2] + pL) / (rhoL + 1e-10)
        hR = (self.to_conservative(WR)[2] + pR) / (rhoR + 1e-10)
        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + 1e-10)
        h_roe = (hL * np.sqrt(rhoL) + hR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + 1e-10)
        c_roe = np.sqrt((self.gamma - 1) * (h_roe - 0.5 * u_roe**2))
        return np.array([rho_roe, u_roe, h_roe, c_roe])
    
    def roe_eigenstructure(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        rho_roe, u_roe, h_roe, c_roe = self.roe_averaged_state(WL, WR)
        eigenvalues = np.array([u_roe - c_roe, u_roe, u_roe + c_roe])
        eigenvectors = [
            np.array([1, u_roe - c_roe, h_roe - u_roe * c_roe]),
            np.array([1, u_roe, 0.5 * u_roe**2]),
            np.array([1, u_roe + c_roe, h_roe + u_roe * c_roe])
        ]
        delta = 0.1 * c_roe
        return eigenvalues, eigenvectors, delta
    
    def roe_wave_strengths(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        rho_roe, u_roe, h_roe, c_roe = self.roe_averaged_state(WL, WR)
        delta_U = UR - UL
        delta_rho = delta_U[0]
        delta_rho_u = delta_U[1]
        delta_rho_E = delta_U[2]
        alpha_2 = ((self.gamma - 1) / (c_roe**2 + 1e-10)) * (
            delta_rho * (0.5 * u_roe**2 - h_roe) + delta_rho_u * u_roe + delta_rho_E
        )
        alpha_1 = ((delta_rho - alpha_2) * (u_roe + c_roe) - delta_rho_u) / (2 * c_roe + 1e-10)
        alpha_3 = (delta_rho_u - (delta_rho - alpha_2) * (u_roe - c_roe)) / (2 * c_roe + 1e-10)
        return np.array([alpha_1, alpha_2, alpha_3])


class IsentropicGasSystem(EquationSystem):
    def __init__(self, gamma: float = 1.4, k: float = 1.0, rho_min: float = 1e-10):
        super().__init__(rho_min)
        self.gamma = gamma
        self.k = k
        self.velocity_index = 1
        self.monitored_index = 0
        self.safeguarded_indices = [0]
    
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        rho, u = W
        return np.array([rho, rho * u])
    
    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        rho, rho_u = U
        u = rho_u / (rho + self.min_var)
        return np.array([rho, u])
    
    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        rho, u = W
        p = self.k * rho**self.gamma
        return np.array([rho * u, rho * u**2 + p])
    
    def sound_speed(self, W: np.ndarray) -> float:
        rho = W[0]
        return np.sqrt(self.gamma * self.k * rho**(self.gamma - 1))
    
    def get_variable_names(self) -> list:
        return ['density', 'velocity']
    
    def hllc_wave_speeds(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        rhoL, uL = WL
        rhoR, uR = WR
        cL = self.sound_speed(WL)
        cR = self.sound_speed(WR)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = 0.5 * (uL + uR) if abs(denom) < 1e-10 else (
            pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)
        ) / denom
        return S_L, S_R, S_star
    
    def hllc_intermediate_states(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray, S_L: float, S_R: float, S_star: float) -> tuple:
        rhoL, uL = WL
        rhoR, uR = WR
        rhoL_star = max(rhoL * (S_L - uL) / (S_L - S_star + 1e-10), self.min_var)
        rhoR_star = max(rhoR * (S_R - uR) / (S_R - S_star + 1e-10), self.min_var)
        UL_star = np.array([rhoL_star, rhoL_star * S_star])
        UR_star = np.array([rhoR_star, rhoR_star * S_star])
        return UL_star, UR_star

    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        rhoL, uL = WL
        rhoR, uR = WR
        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + 1e-10)
        c_roe = self.sound_speed(np.array([rho_roe, u_roe]))
        return np.array([rho_roe, u_roe, c_roe])
    
    def roe_eigenstructure(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        rho_roe, u_roe, c_roe = self.roe_averaged_state(WL, WR)
        eigenvalues = np.array([u_roe - c_roe, u_roe + c_roe])
        eigenvectors = [
            np.array([1, u_roe - c_roe]),
            np.array([1, u_roe + c_roe])
        ]
        delta = 0.1 * c_roe
        return eigenvalues, eigenvectors, delta
    
    def roe_wave_strengths(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        rho_roe, u_roe, c_roe = self.roe_averaged_state(WL, WR)
        delta_U = UR - UL
        alpha_2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe + 1e-10)
        alpha_1 = delta_U[0] - alpha_2
        return np.array([alpha_1, alpha_2])
