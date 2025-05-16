import numpy as np
import matplotlib.pyplot as plt
from .equation import EquationSystem
from .fluxes import Flux
from .reconstructions import Reconstruction


class Solver:
    """Godunov-type solver for hyperbolic conservation laws.

    Solves 1D systems using specified flux and reconstruction methods.
    """

    def __init__(self, equation_system: EquationSystem, flux: str = 'HLLC', reconstruction: str = 'weno5',
                 cfl: float = 0.5, bc_type: str = 'periodic'):
        """Initialize the solver.

        Args:
            equation_system (EquationSystem): The equation system to solve.
            flux (str): Flux method ('lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe').
            reconstruction (str): Reconstruction method ('piecewise_constant', 'muscl', 'ppm', 'weno5').
            cfl (float): Courant-Friedrichs-Lewy number (default: 0.5).
            bc_type (str): Boundary condition type ('periodic', 'reflective', 'transmissive').

        Raises:
            TypeError: If equation_system is not an EquationSystem instance.
            ValueError: If flux, reconstruction, or bc_type is unsupported.
        """
        if not isinstance(equation_system, EquationSystem):
            raise TypeError("equation_system must be an instance of EquationSystem")
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
        self.bc_type = bc_type

        # Boundary condition handlers
        self.bc_handlers = {
            'periodic': self.apply_periodic_bc,
            'reflective': self.apply_reflective_bc,
            'transmissive': self.apply_transmissive_bc
        }
        if bc_type not in self.bc_handlers:
            raise ValueError(f"Unsupported boundary condition: {bc_type}. Choose from {list(self.bc_handlers.keys())}")

    def compute_dt(self, U: np.ndarray, dx: float) -> float:
        """Compute time step based on CFL condition.

        dt = CFL * dx / max_speed

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            dx (float): Spatial grid spacing.

        Returns:
            float: Time step size.
        """
        n_cells = U.shape[1]
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n_cells)]).T
        # Maximum wave speed: |velocity| + sound speed
        max_speed = np.max(np.abs(W[self.equation_system.velocity_index]) + self.equation_system.sound_speed(W))
        # Adjust CFL for high-order methods
        adaptive_cfl = min(self.cfl, 0.4 if self.reconstruction_method.__name__ == 'weno5' else self.cfl)
        return adaptive_cfl * dx / (max_speed + 1e-10)

    def apply_periodic_bc(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply periodic boundary conditions.

        Copies states from opposite ends to ghost cells.

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            n_ghost (int): Number of ghost cells per side.

        Returns:
            np.ndarray: Extended state array with ghost cells.
        """
        n_vars, n_cells = U.shape
        U_ext = np.zeros((n_vars, n_cells + 2 * n_ghost))
        U_ext[:, n_ghost:-n_ghost] = U
        for i in range(n_ghost):
            U_ext[:, i] = U[:, -n_ghost + i]  # Left ghost cells
            U_ext[:, -n_ghost + i] = U[:, i]  # Right ghost cells
        return U_ext

    def apply_reflective_bc(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply reflective boundary conditions.

        Mirrors states and negates velocity at boundaries.

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            n_ghost (int): Number of ghost cells per side.

        Returns:
            np.ndarray: Extended state array with ghost cells.
        """
        n_vars, n_cells = U.shape
        U_ext = np.zeros((n_vars, n_cells + 2 * n_ghost))
        U_ext[:, n_ghost:-n_ghost] = U
        velocity_idx = self.equation_system.velocity_index
        for i in range(n_ghost):
            # Left boundary
            U_ext[:, i] = U[:, n_ghost - 1 - i]
            if velocity_idx is not None:
                U_ext[velocity_idx, i] = -U[velocity_idx, n_ghost - 1 - i]
            # Right boundary
            U_ext[:, -n_ghost + i] = U[:, -i - 1]
            if velocity_idx is not None:
                U_ext[velocity_idx, -n_ghost + i] = -U[velocity_idx, -i - 1]
        return U_ext

    def apply_transmissive_bc(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply transmissive boundary conditions.

        Copies boundary states to ghost cells (zero gradient).

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            n_ghost (int): Number of ghost cells per side.

        Returns:
            np.ndarray: Extended state array with ghost cells.
        """
        n_vars, n_cells = U.shape
        U_ext = np.zeros((n_vars, n_cells + 2 * n_ghost))
        U_ext[:, n_ghost:-n_ghost] = U
        for i in range(n_ghost):
            U_ext[:, i] = U[:, 0]  # Left ghost cells
            U_ext[:, -n_ghost + i] = U[:, -1]  # Right ghost cells
        return U_ext

    def apply_boundary_conditions(self, U: np.ndarray, n_ghost: int) -> np.ndarray:
        """Apply the selected boundary condition.

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            n_ghost (int): Number of ghost cells per side.

        Returns:
            np.ndarray: Extended state array with ghost cells.
        """
        return self.bc_handlers[self.bc_type](U, n_ghost)

    def solve(self, U0: np.ndarray, x: np.ndarray, T: float, n_ghost: int = 2) -> tuple:
        """Solve the hyperbolic system until time T.

        Uses Godunov-type method with specified flux and reconstruction.

        Args:
            U0 (np.ndarray): Initial conservative variables [n_vars, n_cells].
            x (np.ndarray): Spatial grid points.
            T (float): Final simulation time.
            n_ghost (int): Number of ghost cells per side (default: 2).

        Returns:
            tuple: Solution history (array of states) and final time.
        """
        dx = x[1] - x[0]
        U = U0.copy()
        t = 0.0
        history = [U.copy()]
        while t < T:
            dt = min(self.compute_dt(U, dx), T - t)
            # Apply boundary conditions
            U_ext = self.apply_boundary_conditions(U, n_ghost)
            W_ext = np.array([self.equation_system.to_primitive(U_ext[:, i]) for i in range(U_ext.shape[1])]).T
            # Reconstruct states
            UL, UR = self.reconstruction_method(U_ext, dx)
            # Compute fluxes
            F = np.zeros_like(UL)
            for i in range(F.shape[1]):
                F[:, i] = self.flux(UL[:, i], UR[:, i], W_ext[:, i + n_ghost - 1], W_ext[:, i + n_ghost])
            # Update solution
            for i in range(U.shape[1]):
                U[:, i] -= dt / dx * (F[:, i + 1] - F[:, i])
            t += dt
            history.append(U.copy())
        return np.array(history), t

    def plot_solution(self, U_history: np.ndarray, x: np.ndarray, T: float, filename: str = None,
                     plot_indices: list = None):
        """Plot the solution evolution for selected variables.

        Args:
            U_history (np.ndarray): Solution history [n_steps, n_vars, n_cells].
            x (np.ndarray): Spatial grid points.
            T (float): Final simulation time.
            filename (str, optional): File to save the plot.
            plot_indices (list, optional): Indices of variables to plot.
        """
        plot_indices = plot_indices or list(range(U_history.shape[1]))
        fig, axs = plt.subplots(len(plot_indices), 1, figsize=(10, 3 * len(plot_indices)))
        if len(plot_indices) == 1:
            axs = [axs]
        for idx, var_idx in enumerate(plot_indices):
            # Plot every 10th step to reduce clutter
            for U in U_history[::max(1, len(U_history) // 10)]:
                W = self.equation_system.to_primitive(U[:, var_idx])
                axs[idx].plot(x, W, label=f't={U_history.shape[0] * T / len(U_history):.2f}')
            axs[idx].set_xlabel('x')
            axs[idx].set_ylabel(self.variable_names[var_idx])
            axs[idx].legend()
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()