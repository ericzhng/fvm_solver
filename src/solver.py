import numpy as np
import matplotlib.pyplot as plt
from .reconstructions import Reconstruction
from .fluxes import Flux

class GodunovSolver:
    """Godunov-type solver for hyperbolic conservation laws."""
    
    def __init__(self, equation_system, flux_method: str = 'HLLC', limiter: str = 'minmod',
                 reconstruction_method: str = 'muscl', cfl: float = 0.5, bc_type: str = 'transmissive'):
        """Initialize solver with numerical methods and parameters.
        
        Args:
            equation_system: Equation system instance
            flux_method: Numerical flux method
            limiter: Slope limiter for MUSCL
            reconstruction_method: State reconstruction method
            cfl: CFL number (0 < cfl <= 1)
            bc_type: Boundary condition type
        """
        self.equation_system = equation_system
        self.cfl = cfl
        self.bc_type = bc_type
        self.reconstruction = Reconstruction(equation_system, limiter)
        self.flux = Flux(equation_system).get_flux(flux_method)
        self.reconstruction_method = reconstruction_method

    def compute_cfl_dt(self, U: np.ndarray, dx: float) -> float:
        """Compute time step based on CFL condition.
        
        Args:
            U: State array (n_vars, n)
            dx: Spatial step size
            
        Returns:
            Time step size
        """
        max_speed = 0
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        for i in range(U.shape[1]):
            W = self.equation_system.to_primitive(U[:, i])
            if W[0] <= min_var:
                continue
            c = self.equation_system.sound_speed(W)
            max_speed = max(max_speed, abs(W[1]) + c)
        max_speed = max(max_speed, 1e-10)
        return self.cfl * dx / max_speed

    def apply_boundary_conditions(self, U: np.ndarray) -> np.ndarray:
        """Apply boundary conditions to state array.
        
        Args:
            U: State array (n_vars, n)
            
        Returns:
            Extended state array (n_vars, n+4)
        """
        n, n_vars = U.shape[1], U.shape[0]
        U_ext = np.zeros((n_vars, n + 4))
        U_ext[:, 2:-2] = U
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        if self.bc_type == 'transmissive':
            U_ext[:, 0:2] = U[:, 0:1]
            U_ext[:, -2:] = U[:, -1:]
        elif self.bc_type == 'reflective':
            U_ext[:, 0:2] = U[:, 0:1]
            U_ext[1, 0:2] = -U_ext[1, 0:2]
            U_ext[:, -2:] = U[:, -1:]
            U_ext[1, -2:] = -U_ext[1, -2:]
        elif self.bc_type == 'periodic':
            U_ext[:, 0:2] = U[:, -2:]
            U_ext[:, -2:] = U[:, 0:2]
        else:
            raise ValueError(f"Unsupported boundary condition: {self.bc_type}")
        for i in range(n + 4):
            U_ext[0, i] = max(U_ext[0, i], min_var)
        return U_ext

    def solve(self, U: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """Advance solution one time step using Godunov scheme.
        
        Args:
            U: Current state array (n_vars, n)
            dx: Spatial step size
            dt: Time step size
            
        Returns:
            Updated state array
        """
        n, n_vars = U.shape[1], U.shape[0]
        flux = np.zeros((n_vars, n + 1))
        U_new = np.zeros_like(U)
        U_ext = self.apply_boundary_conditions(U)
        
        # Reconstruct states
        reconstructions = {
            'piecewise_constant': self.reconstruction.piecewise_constant,
            'muscl': self.reconstruction.muscl,
            'ppm': self.reconstruction.ppm,
            'weno5': self.reconstruction.weno5
        }
        if self.reconstruction_method not in reconstructions:
            raise ValueError(f"Unsupported reconstruction: {self.reconstruction_method}")
        UL, UR = reconstructions[self.reconstruction_method](U_ext, dx)
        
        # Compute fluxes
        for i in range(n + 1):
            flux[:, i] = self.flux(UL[:, i + 1], UR[:, i + 1])
        
        # Update states
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        for i in range(n):
            U_new[:, i] = U[:, i] - dt / dx * (flux[:, i + 1] - flux[:, i])
            U_new[0, i] = max(U_new[0, i], min_var)
        return U_new

    def plot_variable_evolution(self, snapshots: list, times: list, x: np.ndarray) -> None:
        """Plot evolution of the first variable over time.
        
        Args:
            snapshots: List of state snapshots
            times: List of corresponding times
            x: Spatial grid
        """
        var_name = self.equation_system.get_variable_names()[0]
        plt.figure(figsize=(10, 6))
        for t, var in zip(times, snapshots):
            plt.plot(x, var, label=f't = {t:.3f}', alpha=0.7)
        plt.title(f'{var_name} Evolution ({self.equation_system.__class__.__name__}, '
                 f'{self.reconstruction_method}, {self.flux.__name__})')
        plt.xlabel('x')
        plt.ylabel(var_name)
        plt.grid(True)
        plt.legend()
        plt.savefig(f'variable_evolution_{self.equation_system.__class__.__name__}_'
                   f'{self.reconstruction_method}_{self.flux.__name__}.png')
        plt.close()