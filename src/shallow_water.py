import numpy as np
from .equation import EquationSystem


class ShallowWaterSystem(EquationSystem):
    """Shallow water equations for 1D flows.

    Models conservation of water height and momentum with gravitational effects.

    Attributes:
        g (float): Gravitational acceleration (m/s^2).
        min_var (float): Minimum height for numerical stability.
        velocity_index (int): Index of velocity in primitive variables.
        monitored_index (int): Index of monitored variable (height).
        safeguarded_indices (list): Indices requiring minimum value checks.
        variable_names (list): Names of primitive variables.
    """

    def __init__(self, g: float = 9.81):
        """Initialize the shallow water system.

        Args:
            g (float): Gravitational acceleration (default: 9.81 m/s^2).

        Raises:
            ValueError: If g <= 0.
        """
        if g <= 0:
            raise ValueError("g must be positive")
        super().__init__()
        self.g = g
        self.min_var = 1e-10
        self.velocity_index = 1
        self.monitored_index = 0
        self.safeguarded_indices = [0]
        self.variable_names = ['height', 'velocity']
        self.n_vars = 2

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive to conservative variables.

        Primitive: [height, velocity]
        Conservative: [height, momentum = height * velocity]

        Args:
            W (np.ndarray): Primitive variables, shape (2,).

        Returns:
            np.ndarray: Conservative variables, shape (2,).

        Raises:
            ValueError: If W shape is invalid.
        """
        if W.shape != (self.n_vars,):
            raise ValueError(f"W must have shape ({self.n_vars},)")
        h, u = W
        h = np.maximum(h, self.min_var)
        return np.array([h, h * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative to primitive variables.

        Args:
            U (np.ndarray): Conservative variables, shape (2,).

        Returns:
            np.ndarray: Primitive variables, shape (2,).

        Raises:
            ValueError: If U shape is invalid.
        """
        if U.shape != (self.n_vars,):
            raise ValueError(f"U must have shape ({self.n_vars},)")
        h, hu = U
        h = np.maximum(h, self.min_var)
        return np.array([h, hu / h])

    def sound_speed(self, W: np.ndarray) -> float:
        """Compute wave speed for shallow water equations.

        Formula: sqrt(g * h)

        Args:
            W (np.ndarray): Primitive variables [height, velocity].

        Returns:
            float: Wave speed.

        Raises:
            ValueError: If W shape is invalid.
        """
        if W.shape != (self.n_vars,):
            raise ValueError(f"W must have shape ({self.n_vars},)")
        h = np.maximum(W[0], self.min_var)
        return np.sqrt(self.g * h)

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute physical flux for shallow water equations.

        Flux: [h*u, h*u^2 + 0.5*g*h^2]

        Args:
            U (np.ndarray): Conservative variables [height, momentum].
            W (np.ndarray): Primitive variables [height, velocity].

        Returns:
            np.ndarray: Flux vector, shape (2,).

        Raises:
            ValueError: If U or W shape is invalid.
        """
        if U.shape != (self.n_vars,) or W.shape != (self.n_vars,):
            raise ValueError(f"U and W must have shape ({self.n_vars},)")
        h, u = W
        h = np.maximum(h, self.min_var)
        return np.array([h * u, h * u**2 + 0.5 * self.g * h**2])

    def hllc_states_and_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute HLLC wave speeds, intermediate states, and flux for shallow water equations.

        Combines wave speed and intermediate state calculations for HLLC flux.

        Args:
            WL (np.ndarray): Left primitive state [height, velocity].
            WR (np.ndarray): Right primitive state [height, velocity].
            UL (np.ndarray): Left conservative state [height, momentum].
            UR (np.ndarray): Right conservative state [height, momentum].

        Returns:
            tuple: (S_L, S_R, S_star, UL_star, UR_star, F), where:
                - S_L, S_R, S_star: Left, right, and contact wave speeds.
                - UL_star, UR_star: Left and right intermediate conservative states.
                - F: HLLC numerical flux.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if any(arr.shape != (self.n_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.n_vars},)")
        
        # Extract variables
        hL, uL = WL
        hR, uR = WR
        hL = np.maximum(hL, self.min_var)
        hR = np.maximum(hR, self.min_var)
        
        # Compute wave speeds
        cL = np.sqrt(self.g * hL)
        cR = np.sqrt(self.g * hR)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        denom = hR * (S_R - uR) - hL * (S_L - uL)
        S_star = (hR * uR * (S_R - uR) - hL * uL * (S_L - uL) + 0.5 * self.g * (hR**2 - hL**2)) / (denom + self.min_var)
        
        # Compute intermediate states
        hL_star = np.maximum(hL * (S_L - uL) / (S_L - S_star + self.min_var), self.min_var)
        hR_star = np.maximum(hR * (S_R - uR) / (S_R - S_star + self.min_var), self.min_var)
        UL_star = np.array([hL_star, hL_star * S_star])
        UR_star = np.array([hR_star, hR_star * S_star])
        
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
            
        return S_L, S_R, S_star, UL_star, UR_star, F
    
    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Compute Roe-averaged state variables.

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray: [h_roe, u_roe, c_roe].
        """
        if any(arr.shape != (self.n_vars,) for arr in [WL, WR]):
            raise ValueError(f"WL and WR must have shape ({self.n_vars},)")
        hL, uL = WL
        hR, uR = WR
        hL = np.maximum(hL, self.min_var)
        hR = np.maximum(hR, self.min_var)
        h_roe = np.sqrt(hL * hR)
        u_roe = (uL * np.sqrt(hL) + uR * np.sqrt(hR)) / (np.sqrt(hL) + np.sqrt(hR) + self.min_var)
        c_roe = np.sqrt(self.g * h_roe)
        return np.array([h_roe, u_roe, c_roe])

    def roe_eigenstructure(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute Roe eigenstructure for shallow water equations.

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.

        Returns:
            tuple: Eigenvalues, eigenvectors, and entropy fix parameter (delta).
        """
        if any(arr.shape != (self.n_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.n_vars},)")
        h_roe, u_roe, c_roe = self.roe_averaged_state(WL, WR)
        eigenvalues = np.array([u_roe - c_roe, u_roe + c_roe])
        eigenvectors = [
            np.array([1, u_roe - c_roe]),
            np.array([1, u_roe + c_roe])
        ]
        delta = 0.1 * c_roe
        return eigenvalues, eigenvectors, delta

    def roe_wave_strengths(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute Roe wave strengths for shallow water equations.

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Wave strength coefficients.
        """
        if any(arr.shape != (self.n_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.n_vars},)")
        h_roe, u_roe, c_roe = self.roe_averaged_state(WL, WR)
        delta_U = UR - UL
        alpha_2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe + self.min_var)
        alpha_1 = delta_U[0] - alpha_2
        return np.array([alpha_1, alpha_2])

    def roe_states_and_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute Roe-averaged state, eigenstructure, wave strengths, and flux for shallow water equations.

        Combines Roe-related calculations for Roe flux with entropy fix.

        Args:
            WL (np.ndarray): Left primitive state [height, velocity].
            WR (np.ndarray): Right primitive state [height, velocity].
            UL (np.ndarray): Left conservative state [height, momentum].
            UR (np.ndarray): Right conservative state [height, momentum].

        Returns:
            tuple: (h_roe, u_roe, c_roe, eigenvalues, eigenvectors, delta, wave_strengths, F), where:
                - h_roe, u_roe, c_roe: Roe-averaged height, velocity, sound speed.
                - eigenvalues: Roe eigenvalues.
                - eigenvectors: Roe eigenvectors.
                - delta: Entropy fix parameter.
                - wave_strengths: Wave strength coefficients.
                - F: Roe numerical flux.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if any(arr.shape != (self.n_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.n_vars},)")
        
        # Roe-averaged state
        hL, uL = WL
        hR, uR = WR
        hL = np.maximum(hL, self.min_var)
        hR = np.maximum(hR, self.min_var)
        h_roe = np.sqrt(hL * hR)
        u_roe = (uL * np.sqrt(hL) + uR * np.sqrt(hR)) / (np.sqrt(hL) + np.sqrt(hR) + self.min_var)
        c_roe = np.sqrt(self.g * h_roe)
        
        # Eigenstructure
        eigenvalues = np.array([u_roe - c_roe, u_roe + c_roe])
        eigenvectors = [
            np.array([1, u_roe - c_roe]),
            np.array([1, u_roe + c_roe])
        ]
        delta = 0.1 * c_roe
        
        # Wave strengths
        delta_U = UR - UL
        alpha_2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe + self.min_var)
        alpha_1 = delta_U[0] - alpha_2
        wave_strengths = np.array([alpha_1, alpha_2])
        
        # Compute flux
        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)
        abs_A = np.zeros_like(FL)
        for i in range(len(eigenvalues)):
            abs_lambda = abs(eigenvalues[i]) if abs(eigenvalues[i]) > delta else (eigenvalues[i]**2 + delta**2) / (2 * delta)
            abs_A += abs_lambda * wave_strengths[i] * eigenvectors[i]
        F = 0.5 * (FL + FR - abs_A)
        
        return h_roe, u_roe, c_roe, eigenvalues, eigenvectors, delta, wave_strengths, F
    