import numpy as np
from ..src.equation import EquationSystem

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

    def roe_averaged_state(self, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
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
    