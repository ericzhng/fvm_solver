import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Dict
from .reconstructions import Reconstruction
from .equation import EquationSystem
from .fluxes import Flux

class GodunovSolver:
    """Godunov-type solver for hyperbolic conservation laws with flexible variable ordering."""
    
    def __init__(self, equation_system: EquationSystem, flux: str = 'HLLC', reconstruction: str = 'weno5', cfl: float = 0.5, boundary_type: str = 'periodic'):
        """Initialize solver with equation system, flux, reconstruction, CFL number, and boundary type."""
        self.equation_system = equation_system
        self.flux = Flux(equation_system, lambda_max=1.0).get_flux(flux)
        self.reconstruction = Reconstruction(equation_system, limiter='minmod' if reconstruction == 'muscl' else None)
        self.reconstruction_method = {
            'piecewise_constant': self.reconstruction.piecewise_constant,
            'muscl': self.reconstruction.muscl,
            'ppm': self.reconstruction.ppm,
            'weno5': self.reconstruction.weno5
        }[reconstruction]
        self.cfl = cfl
        self.variable_names = equation_system.get_variable_names()
        self.boundary_type = boundary_type
        
        # Define boundary condition handlers
        self.bc_handlers = {
            'periodic': self.apply_periodic_bc,
            'reflective': self.apply_reflective_bc,
            'outflow': self.apply_outflow_bc,
            'transmissive': self.apply_outflow_bc  # Transmissive is equivalent to outflow in 1D
        }
        
        if boundary_type not in self.bc_handlers:
            raise ValueError(f"Unsupported boundary type: {boundary_type}. Choose from {list(self.bc_handlers.keys())}")
    
    def compute_dt(self, U: np.ndarray, dx: float) -> float:
        """Compute time step based on CFL condition."""
        n = U.shape[1]
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n)]).T
        max_speed = np.max(np.abs(W[self.equation_system.velocity_index]) + self.equation_system.sound_speed(W))
        adaptive_cfl = min(self.cfl, 0.4 if self.reconstruction_method.__name__ == 'weno5' else self.cfl)
        return adaptive_cfl * dx / (max_speed + 1e-10)
    
    def apply_periodic_bc(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply periodic boundary conditions."""
        U_ext = np.zeros((U.shape[0], U.shape[1] + 2 * n_ghost))
        U_ext[:, n_ghost:-n_ghost] = U
        for i in range(n_ghost):
            U_ext[:, i] = U[:, -n_ghost + i]
            U_ext[:, -n_ghost + i] = U[:, i]
        return U_ext
    
    def apply_reflective_bc(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply reflective boundary conditions (mirror velocity)."""
        U_ext = np.zeros((U.shape[0], U.shape[1] + 2 * n_ghost))
        U_ext[:, n_ghost:-n_ghost] = U
        velocity_idx = self.equation_system.velocity_index
        for i in range(n_ghost):
            # Left boundary: copy variables, negate velocity
            U_ext[:, i] = U[:, n_ghost - 1 - i]
            if velocity_idx is not None:
                U_ext[velocity_idx, i] = -U[velocity_idx, n_ghost - 1 - i]
            # Right boundary: copy variables, negate velocity
            U_ext[:, -n_ghost + i] = U[:, -i - 1]
            if velocity_idx is not None:
                U_ext[velocity_idx, -n_ghost + i] = -U[velocity_idx, -i - 1]
        return U_ext
    
    def apply_outflow_bc(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply outflow/transmissive boundary conditions (zero gradient)."""
        U_ext = np.zeros((U.shape[0], U.shape[1] + 2 * n_ghost))
        U_ext[:, n_ghost:-n_ghost] = U
        for i in range(n_ghost):
            # Left boundary: copy leftmost interior values
            U_ext[:, i] = U[:, 0]
            # Right boundary: copy rightmost interior values
            U_ext[:, -n_ghost + i] = U[:, -1]
        return U_ext
    
    def apply_boundary_conditions(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply selected boundary conditions using bc_handlers."""
        U_ext = self.bc_handlers[self.boundary_type](U, n_ghost)
        for idx in self.safeguarded_indices:
            U_ext[idx] = np.maximum(U_ext[idx], self.min_var)
        return U_ext
    
    def solve(self, U0: np.ndarray, x: np.ndarray, T: float, n_ghost: int = 2) -> tuple:
        """Solve the hyperbolic system until time T."""
        dx = x[1] - x[0]
        U = U0.copy()
        t = 0.0
        history = [U.copy()]
        
        while t < T:
            dt = min(self.compute_dt(U, dx), T - t)
            U_ext = self.apply_boundary_conditions(U, n_ghost)
            W_ext = np.array([self.equation_system.to_primitive(U_ext[:, i]) for i in range(U_ext.shape[1])]).T
            UL, UR = self.reconstruction_method(U_ext, dx)
            
            F = np.zeros_like(UL)
            for i in range(F.shape[1]):
                F[:, i] = self.flux(UL[:, i], UR[:, i], W_ext[:, i + n_ghost - 1], W_ext[:, i + n_ghost])
            
            for i in range(U.shape[1]):
                U[:, i] -= dt / dx * (F[:, i + 1] - F[:, i])
            
            t += dt
            history.append(U.copy())
        
        return np.array(history), t
    
    def plot_solution(self, U_history: np.ndarray, x: np.ndarray, T: float, filename: str = None, plot_indices: list = None):
        """Plot the solution evolution for selected variables."""
        plot_indices = plot_indices or list(range(U_history.shape[1]))
        fig, axs = plt.subplots(len(plot_indices), 1, figsize=(10, 3 * len(plot_indices)))
        if len(plot_indices) == 1:
            axs = [axs]
        
        for idx, i in enumerate(plot_indices):
            for U in U_history[::max(1, len(U_history) // 10)]:
                W = self.equation_system.to_primitive(U[:, i])
                axs[idx].plot(x, W, label=f't={U_history.shape[0] * T / len(U_history):.2f}')
            axs[idx].set_xlabel('x')
            axs[idx].set_ylabel(self.variable_names[i])
            axs[idx].legend()
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
