import numpy as np
from .base_equation import EquationSystem


# for transport equation
class CustomSystem(EquationSystem):

    def __init__(self, param1, min_var=1e-10):
        super().__init__(min_var)
        self.param1 = param1
        self.velocity_index = 1
        self.monitored_index = 0
        self.safeguarded_indices = [0]
    
    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        # Example
        return np.array([W[0], W[0] * W[1]])
    
    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        # Example
        return np.array([U[0], U[1] / (U[0] + self.min_var)])
    
    def sound_speed(self, W: np.ndarray) -> float:
        """Compute sound speed numerically via Jacobian eigenvalues."""
        # Convert to conservative state
        U = self.to_conservative(W)
        n_vars = len(U)
        epsilon = 1e-6
        A = np.zeros((n_vars, n_vars))
        
        # Numerical Jacobian
        for j in range(n_vars):
            U_pert = U.copy()
            U_pert[j] += epsilon
            W_pert = self.to_primitive(U_pert)
            F_plus = self.compute_flux(U_pert, W_pert)
            F = self.compute_flux(U, W)
            A[:, j] = (F_plus - F) / epsilon
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        # Sound speed: max eigenvalue magnitude, adjusted for velocity
        u = W[self.velocity_index]
        return np.max(np.abs(eigenvalues - u))
    
    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        # Example flux (replace with actual system)
        return np.array([W[0] * W[1], W[1]**2 + self.param1 * W[0]])
    
    def get_variable_names(self) -> list:
        return ['var1', 'velocity']

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

    def hllc_states_and_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute HLLC wave speeds, intermediate states, and flux (numerical default).

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.

        Returns:
            tuple: (S_L, S_R, S_star, UL_star, UR_star, F), where:
                - S_L, S_R, S_star: Left, right, and contact wave speeds.
                - UL_star, UR_star: Left and right intermediate conservative states.
                - F: HLLC numerical flux.
        """
        if any(arr.shape != (self.n_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.n_vars},)")

        S_L, S_R, S_star = self.hllc_wave_speeds(WL, WR, UL, UR)
        UL_star, UR_star = self.hllc_intermediate_states(WL, WR, UL, UR, S_L, S_R, S_star)
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

    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray, use_jacobian: bool = False) -> np.ndarray:
        """Compute Roe-averaged state variables.

        Typically use parameter vector for averaging, usually analytically.
        If no parameter vector is known, then use Arithmetic Mean.
        Sometimes use primitive variable averaging, like what's used here.
        Another way is to use numerical Jacobian to compute Roe-averaged state.

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray:  u_roe, c_roe.
        """
        if use_jacobian:
            UL = self.to_conservative(WL)
            UR = self.to_conservative(UR)
            U_roe = 0.5 * (UL + UR)
            n_vars = len(UL)
            A_roe = np.zeros((n_vars, n_vars))
            epsilon = 1e-6
            for j in range(n_vars):
                U_pert = U_roe.copy()
                U_pert[j] += epsilon
                W_pert = self.to_primitive(U_pert)
                F_plus = self.compute_flux(U_pert, W_pert)
                F = self.compute_flux(U_roe, self.to_primitive(U_roe))
                A_roe[:, j] = (F_plus - F) / epsilon
            eigenvalues = np.linalg.eigvals(A_roe)
            u_roe = self.to_primitive(U_roe)[self.velocity_index]
            c_roe = max(abs(eigenvalues - u_roe))
            return np.array([u_roe, c_roe])
        else:
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

    def roe_states_and_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> tuple:
        """Compute Roe-averaged state, eigenstructure, wave strengths, and flux (numerical default).

        Args:
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.

        Returns:
            tuple: (var1_roe, var2_roe, c_roe, eigenvalues, eigenvectors, delta, wave_strengths, F), where:
                - var1_roe, var2_roe: Roe-averaged variables (e.g., u_roe for velocity, h_roe or rho_roe).
                - c_roe: Roe-averaged sound speed.
                - eigenvalues: Roe eigenvalues.
                - eigenvectors: Roe eigenvectors.
                - delta: Entropy fix parameter.
                - wave_strengths: Wave strength coefficients.
                - F: Roe numerical flux.
        """
        if any(arr.shape != (self.n_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.n_vars},)")

        # Roe-averaged state
        u_roe, c_roe = self.roe_averaged_state(WL, WR)
        var1_roe = u_roe  # For consistency, could be velocity or another variable
        var2_roe = None  # Placeholder, to be overridden by subclasses (e.g., h_roe for shallow water)

        # Eigenstructure and wave strengths
        eigenvalues, eigenvectors, delta = self.roe_eigenstructure(WL, WR, UL, UR)
        wave_strengths = self.roe_wave_strengths(WL, WR, UL, UR)

        # Compute flux
        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)
        abs_A = np.zeros_like(FL)
        for i in range(len(eigenvalues)):
            abs_lambda = abs(eigenvalues[i]) if abs(eigenvalues[i]) > delta else (eigenvalues[i]**2 + delta**2) / (2 * delta)
            abs_A += abs_lambda * wave_strengths[i] * eigenvectors[i]
        F = 0.5 * (FL + FR - abs_A)

        return var1_roe, var2_roe, c_roe, eigenvalues, eigenvectors, delta, wave_strengths, F
