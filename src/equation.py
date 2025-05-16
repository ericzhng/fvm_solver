import numpy as np


class EquationSystem:
    """Base class for hyperbolic conservation law equation systems.

    Provides default numerical methods for flux calculations and state conversions.
    Subclasses should override methods for analytical implementations.
    """

    def __init__(self, min_var: float = 1e-10):
        """Initialize equation system with minimum variable threshold."""
        self.min_var = min_var  # Minimum value for numerical stability
        self.velocity_index = None
        self.monitored_index = None
        self.safeguarded_indices : list[int] = []
        self.variable_names : list[str] = []

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Args:
            W (np.ndarray): Primitive variables.

        Returns:
            np.ndarray: Conservative variables.
        """
        raise NotImplementedError

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables.

        Returns:
            np.ndarray: Primitive variables.
        """
        raise NotImplementedError

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute the physical flux for the given state.

        Args:
            U (np.ndarray): Conservative variables.
            W (np.ndarray): Primitive variables.

        Returns:
            np.ndarray: Flux vector.
        """
        raise NotImplementedError

    def sound_speed(self, W: np.ndarray) -> float:
        """Compute the sound speed for the given primitive state.

        Args:
            W (np.ndarray): Primitive variables.

        Returns:
            float: Sound speed.
        """
        raise NotImplementedError

    def get_variable_names(self) -> list:
        """Return the names of the primitive variables.

        Returns:
            list: List of variable names.
        """
        return self.variable_names


    def hllc_wave_speeds(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute HLLC wave speeds (numerical default).

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.

        Returns:
            tuple: Left, right, and contact wave speeds (S_L, S_R, S_star).
        """

        cL = self.sound_speed(WL)
        cR = self.sound_speed(WR)

        uL = WL[self.velocity_index] if self.velocity_index is not None else 0.0
        uR = WR[self.velocity_index] if self.velocity_index is not None else 0.0

        S_L = min(float(uL - cL), float(uR - cR))
        S_R = max(float(uL + cL), float(uR + cR))
        
        # Numerical approximation for S_star
        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)

        S_star = 0.5 * (uL + uR) + (FL[self.velocity_index] - FR[self.velocity_index]) / ( UL[self.velocity_index] - UR[self.velocity_index] + 1e-10 )

        return S_L, S_R, S_star


    def hllc_intermediate_states(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray,
                                S_L: float, S_R: float, S_star: float) -> tuple:
        """Compute HLLC intermediate states (numerical default).

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.
            S_L (float): Left wave speed.
            S_R (float): Right wave speed.
            S_star (float): Contact wave speed.

        Returns:
            tuple: Left and right intermediate conservative states (UL_star, UR_star).
        """

        # Numerical approximation: assume intermediate states based on wave speeds
        UL_star = UL + (S_L * (S_star - WL[self.velocity_index]) / (S_L - S_star + 1e-10)) * (UL - self.compute_flux(UL, WL) / S_L)
        UR_star = UR + (S_R * (S_star - WR[self.velocity_index]) / (S_R - S_star + 1e-10)) * (UR - self.compute_flux(UR, WR) / S_R)

        for idx in self.safeguarded_indices:
            UL_star[idx] = max(UL_star[idx], self.min_var)
            UR_star[idx] = max(UR_star[idx], self.min_var)

        return UL_star, UR_star


    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Compute Roe-averaged state variables numerically (numerical default).

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray:  u_roe, c_roe.
        """
        UL = self.to_conservative(WL)
        UR = self.to_conservative(WR)
        U_roe = 0.5 * (UL + UR)
        W_roe = self.to_primitive(U_roe)
        c_roe = self.sound_speed(W_roe)
        return np.array([W_roe[self.velocity_index], c_roe])
    

    def roe_eigenstructure(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute Roe eigenstructure (numerical default).

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.

        Returns:
            tuple: Eigenvalues, eigenvectors, and entropy fix parameter (delta).
        """
        u_roe, c_roe = self.roe_averaged_state(WL, WR)
        n_vars = len(WL)
        eigenvalues = np.array([u_roe - c_roe, u_roe, u_roe + c_roe][:n_vars])
        eigenvectors = [
            np.ones(n_vars),  # Simplified: assume neutral wave
            np.array([1.0 if i == self.velocity_index else 0.0 for i in range(n_vars)]),  # Velocity wave
            np.ones(n_vars)  # Simplified: assume acoustic wave
        ][:n_vars]
        delta = 0.1 * c_roe
        return eigenvalues, eigenvectors, delta


    def roe_wave_strengths(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute Roe wave strengths (numerical default).

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Wave strength coefficients (alpha).
        """
        n_vars = len(WL)
        delta_U = UR - UL
        # Simple numerical approximation: equal distribution of jump
        alpha = delta_U / (n_vars + 1e-10)
        return alpha[:n_vars]
