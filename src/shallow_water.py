import numpy as np
from .equation import EquationSystem


class ShallowWaterSystem(EquationSystem):
    """Shallow water equation system for 1D flows.

    Models conservation of height and momentum with gravity.
    """

    def __init__(self, g: float = 9.81):
        """Initialize the shallow water system.

        Args:
            g (float): Gravitational acceleration (default: 9.81 m/s^2).
        """
        super().__init__()
        self.g = g  # Gravity constant
        self.min_var = 1e-10  # Minimum height for stability
        self.velocity_index = 1
        self.monitored_index = 0
        self.safeguarded_indices = [0]  # Safeguard height (h)
        self.variable_names = ['height', 'velocity']
    
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Primitive: [height, velocity]
        Conservative: [height, momentum = height * velocity]

        Args:
            W (np.ndarray): Primitive variables [height, velocity].

        Returns:
            np.ndarray: Conservative variables [height, momentum].
        """
        h, u = W
        return np.array([h, h * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables [height, momentum].

        Returns:
            np.ndarray: Primitive variables [height, velocity].
        """
        h, hu = U
        h = np.maximum(h, self.min_var)  # Ensure positive height
        return np.array([h, hu / h])

    def sound_speed(self, W: np.ndarray) -> float:
        """Compute the sound speed (wave speed) for shallow water.

        Args:
            W (np.ndarray): Primitive variables [height, velocity].

        Returns:
            float: Wave speed sqrt(g * h).
        """
        h = np.maximum(W[0], self.min_var)
        return np.sqrt(self.g * h)

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Flux: [h*u, h*u^2 + 0.5*g*h^2]

        Args:
            U (np.ndarray): Conservative variables [height, momentum].
            W (np.ndarray): Primitive variables [height, velocity].

        Returns:
            np.ndarray: Flux vector.
        """
        h, u = W
        h = np.maximum(h, self.min_var)
        return np.array([h * u, h * u**2 + 0.5 * self.g * h**2])

    def hllc_wave_speeds(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute HLLC wave speeds for shallow water.

        Args:
            WL (np.ndarray): Left primitive state [height, velocity].
            WR (np.ndarray): Right primitive state [height, velocity].
            UL (np.ndarray): Left conservative state [height, momentum].
            UR (np.ndarray): Right conservative state [height, momentum].

        Returns:
            tuple: Left, right, and contact wave speeds (S_L, S_R, S_star).
        """
        hL, uL = WL
        hR, uR = WR
        hL = np.maximum(hL, self.min_var)
        hR = np.maximum(hR, self.min_var)
        cL = np.sqrt(self.g * hL)
        cR = np.sqrt(self.g * hR)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)

        denom = hR * (S_R - uR) - hL * (S_L - uL)
        S_star = 0.5 * (uL + uR) if abs(denom) < 1e-10 else (
            hR * uR * (S_R - uR) - hL * uL * (S_L - uL) + 0.5 * self.g * (hR**2 - hL**2)
        ) / denom

        return S_L, S_R, S_star

    def hllc_intermediate_states(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray,
                                S_L: float, S_R: float, S_star: float) -> tuple:
        """Compute HLLC intermediate states for shallow water.

        Args:
            WL (np.ndarray): Left primitive state [height, velocity].
            WR (np.ndarray): Right primitive state [height, velocity].
            UL (np.ndarray): Left conservative state [height, momentum].
            UR (np.ndarray): Right conservative state [height, momentum].
            S_L (float): Left wave speed.
            S_R (float): Right wave speed.
            S_star (float): Contact wave speed.

        Returns:
            tuple: Left and right intermediate conservative states (UL_star, UR_star).
        """
        hL, uL = WL
        hR, uR = WR
        # hL = np.maximum(hL, self.min_var)
        # hR = np.maximum(hR, self.min_var)
        # Intermediate height
        # h_star = 0.5 * (hL + hR)
        # Intermediate states: [h, h*u]
        # UL_star = np.array([h_star, h_star * S_star])
        # UR_star = np.array([h_star, h_star * S_star])

        hL_star = max(hL * (S_L - uL) / (S_L - S_star + self.min_var), self.min_var)
        hR_star = max(hR * (S_R - uR) / (S_R - S_star + self.min_var), self.min_var)

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