import numpy as np
import matplotlib.pyplot as plt
from .equation import EquationSystem
from .fluxes import Flux
from .reconstructions import Reconstruction
from .boundary_conditions import BoundaryCondition


class Solver:
    """Godunov-type solver for 1D hyperbolic conservation laws.

    Integrates systems using specified flux, reconstruction, and boundary conditions.
    """

    def __init__(self, equation_system: EquationSystem, flux: str = 'hllc', reconstruction: str = 'weno5',
                 cfl: float = 0.5, bc_type: str = 'periodic', limiter: str = 'minmod'):
        """Initialize the solver.

        Args:
            equation_system (EquationSystem): The equation system to solve.
            flux (str): Flux method ('lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe', 'roe_general').
            reconstruction (str): Reconstruction method ('piecewise_constant', 'muscl', 'ppm', 'weno5').
            cfl (float): CFL number for time step control (default: 0.5).
            bc_type (str): Boundary condition type ('periodic', 'reflective', 'dirichlet', 'neumann').
            limiter (str): Limiter for MUSCL reconstruction (default: 'minmod').

        Raises:
            TypeError: If equation_system is not an EquationSystem instance.
            ValueError: If flux, reconstruction, or bc_type is unsupported, or cfl <= 0.
        """
        if not isinstance(equation_system, EquationSystem):
            raise TypeError("equation_system must be an EquationSystem instance")
        if cfl <= 0:
            raise ValueError("cfl must be positive")
        self.equation_system = equation_system
        self.flux = Flux(equation_system, lambda_max=1.0).get_flux(flux.lower())
        self.reconstruction = Reconstruction(equation_system, limiter=limiter if reconstruction.lower() == 'muscl' else None)
        self.reconstruction_method = {
            'piecewise_constant': self.reconstruction.piecewise_constant,
            'muscl': self.reconstruction.muscl,
            'ppm': self.reconstruction.ppm,
            'weno5': self.reconstruction.weno5
        }[reconstruction.lower()]
        self.cfl = cfl
        self.bc = BoundaryCondition(equation_system, bc_type.lower())
        self.variable_names = equation_system.get_variable_names()

    def specify_bc(self, left: float, right: float):
        self.bc.left_values = left
        self.bc.right_values = right

    def specify_dx(self, x: np.ndarray):
        dx = x[1] - x[0]
        self.bc.dx = dx

    def compute_dt(self, U: np.ndarray, dx: float) -> float:
        """Compute time step based on CFL condition.

        Formula: dt = CFL * dx / max(|velocity| + sound_speed)

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells).
            dx (float): Spatial grid spacing.

        Returns:
            float: Time step size.

        Raises:
            ValueError: If dx <= 0 or U shape is invalid.
        """
        if dx <= 0:
            raise ValueError("dx must be positive")
        if U.ndim != 2 or U.shape[0] != self.equation_system.n_vars:
            raise ValueError(f"U must have shape ({self.equation_system.n_vars}, n_cells)")
        n_cells = U.shape[1]
        W = self._to_primitive_array(U)
        max_speed = 0.0
        for i in range(n_cells):
            speed = abs(W[self.equation_system.velocity_index, i]) + self.equation_system.sound_speed(W[:, i])
            max_speed = max(max_speed, speed)
        adaptive_cfl = min(
            self.cfl,
            0.4 if self.reconstruction_method.__name__ == 'weno5' else 0.2 if self.reconstruction_method.__name__ == 'ppm' else self.cfl
        )
        return adaptive_cfl * dx / (max_speed + self.equation_system.min_var)

    def _to_primitive_array(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables for all cells.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells).

        Returns:
            np.ndarray: Primitive variables, shape (n_vars, n_cells).
        """
        if U.ndim != 2 or U.shape[0] != self.equation_system.n_vars:
            raise ValueError(f"U must have shape ({self.equation_system.n_vars}, n_cells)")
        n_cells = U.shape[1]
        W = np.zeros_like(U)
        for i in range(n_cells):
            W[:, i] = self.equation_system.to_primitive(U[:, i])
        return W

    def solve(self, U0: np.ndarray, x: np.ndarray, T: float, n_ghost: int = 2) -> tuple:
        """Solve the hyperbolic system until time T.

        Uses Godunov-type method with specified flux and reconstruction.

        Args:
            U0 (np.ndarray): Initial conservative variables, shape (n_vars, n_cells).
            x (np.ndarray): Spatial grid points.
            T (float): Final simulation time.
            n_ghost (int): Number of ghost cells per side (default: 2).

        Returns:
            tuple: (history, t), where history is array of states [n_steps, n_vars, n_cells], t is final time.

        Raises:
            ValueError: If inputs are invalid or n_ghost is insufficient.
        """
        if T < 0:
            raise ValueError("T must be non-negative")
        if x.size < 2:
            raise ValueError("x must have at least 2 points")
        if U0.ndim != 2 or U0.shape[0] != self.equation_system.n_vars:
            raise ValueError(f"U0 must have shape ({self.equation_system.n_vars}, n_cells)")
        min_ghost = 2 if self.reconstruction_method.__name__ in ['ppm', 'weno5'] else 1
        if n_ghost < min_ghost:
            raise ValueError(f"n_ghost must be at least {min_ghost} for {self.reconstruction_method.__name__}")

        dx = x[1] - x[0]
        n_cells = U0.shape[1]
        n_vars = self.equation_system.n_vars
        U = U0.copy()
        t = 0.0
        history = [U.copy()]

        while t < T:
            print(f"Time: {t:.3f}")
            # Apply boundary conditions
            U_ext = self.bc.apply_bcs(U, n_ghost)

            # Compute time step
            dt = min(self.compute_dt(U, dx), T - t)

            # Reconstruct states at interfaces
            WL, WR = self.reconstruction_method(U_ext, n_ghost)
            UL = np.zeros_like(WL)
            UR = np.zeros_like(WR)
            for i in range(WL.shape[1]):
                UL[:, i] = self.equation_system.to_conservative(WL[:, i])
                UR[:, i] = self.equation_system.to_conservative(WR[:, i])

            # Compute fluxes
            F = np.zeros((n_vars, n_cells + 1))
            for i in range(n_cells + 1):
                F[:, i] = self.flux(UL[:, i], UR[:, i], WL[:, i], WR[:, i])

            # Update solution
            U_new = U - (dt / dx) * (F[:, 1:] - F[:, :-1])
            for idx in self.equation_system.safeguarded_indices:
                U_new[idx, :] = np.maximum(U_new[idx, :], self.equation_system.min_var)
            U = U_new

            t += dt

            history.append(U.copy())

        return np.array(history), t

    def plot_solution(self, history: np.ndarray, x: np.ndarray, t: float, variable: str = None):
        """Plot the solution at the final time.

        Args:
            history (np.ndarray): Solution history, shape (n_steps, n_vars, n_cells).
            x (np.ndarray): Spatial grid points.
            t (float): Final time.
            variable (str): Variable to plot (optional, default: all variables).

        Raises:
            ValueError: If variable is invalid or history shape is incorrect.
        """
        xmid = (x[1:] + x[:-1]) / 2
        if history.ndim != 3 or history.shape[1] != self.equation_system.n_vars:
            raise ValueError(f"history must have shape (n_steps, {self.equation_system.n_vars}, n_cells)")
        U_final = history[-1]
        W_final = self._to_primitive_array(U_final)
        idx = None
        if variable:
            if variable not in self.variable_names:
                raise ValueError(f"Variable {variable} not in {self.variable_names}")
            idx = self.variable_names.index(variable)
            plt.plot(xmid, W_final[idx, :], label=variable)
            plt.xlabel('x')
            plt.ylabel(variable)
            plt.title(f'{variable} at t = {t:.3f}')
        else:
            for i, name in enumerate(self.variable_names):
                plt.plot(xmid, W_final[i, :], label=name)
            plt.xlabel('x')
            plt.ylabel('Variables')
            plt.title(f'Solution at t = {t:.3f}')
        plt.legend()
        plt.grid(True)
        plt.show()
