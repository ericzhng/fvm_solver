import numpy as np
import matplotlib.pyplot as plt
from .equation.base_equation import EquationSystem
from .flux import Flux
from .reconstruction import Reconstruction
from .boundary import BoundaryCondition


class Solver:
    """Godunov-type solver for 1D hyperbolic conservation laws.

    Integrates systems using specified flux, reconstruction, and boundary conditions.
    """
    def __init__(self, equation_system: EquationSystem, boundary_condition: BoundaryCondition, 
                 cfl: float = 0.5, flux: str = 'hllc', reconstruction: str = 'weno5', 
                 use_primitive_reconstruction: bool = False, limiter: str = 'minmod'):
        """Initialize the solver.

        Args:
            equation_system (EquationSystem): The equation system to solve.
            reconstruction (str): Reconstruction method ('piecewise_constant', 'muscl', 'ppm', 'weno5').
            flux (str): Flux method ('lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe', 'roe_general').
            cfl (float): CFL number for time step control (default: 0.5).
            bc_type (str): Boundary condition type ('periodic', 'reflective', 'dirichlet', 'neumann').
            limiter (str): Limiter for MUSCL reconstruction (default: 'minmod').

        Raises:
            TypeError: If equation_system is not an EquationSystem instance.
            ValueError: If flux, reconstruction, or bc_type is unsupported, or cfl <= 0.
        """
        if not isinstance(equation_system, EquationSystem):
            raise TypeError("input equation_system must be an EquationSystem instance")
        if cfl <= 0:
            raise ValueError("cfl must be positive")
        
        self.equation_system = equation_system
        self.bc = boundary_condition
        self.flux = Flux(equation_system, lambda_max=1.0).get_flux(flux.lower())
        self.reconstruction = Reconstruction(equation_system, limiter=limiter if reconstruction.lower() == 'muscl' else None)
        self.reconstruction_method = {
            'piecewise_constant': self.reconstruction.piecewise_constant,
            'muscl': self.reconstruction.muscl,
            # 'ppm': self.reconstruction.ppm,
            # 'weno5': self.reconstruction.weno5
        }[reconstruction.lower()]
        self.cfl = cfl
        self.use_primitive_reconstruction = use_primitive_reconstruction
        self.variable_names = equation_system.get_variable_names()

    def compute_dt(self, U: np.ndarray, x: np.ndarray) -> float:
        """Compute time step based on CFL condition.

        Formula: dt = CFL * dx / max(|velocity| + sound_speed)

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells).
            x (float): Spatial grid array.

        Returns:
            float: Time step size.

        Raises:
            ValueError: If dx <= 0 or U shape is invalid.
        """
        if x.size != U.shape[1] + 1:
            raise ValueError("x must be of length n_cells + 1")
        if U.ndim != 2 or U.shape[0] != self.equation_system.n_vars:
            raise ValueError(f"U must have shape ({self.equation_system.n_vars}, n_cells)")
        
        n_cells = U.shape[1]
        W = self._to_primitive_array(U)
        max_value = 0.0
        for i in range(n_cells):
            dx = x[i+1] - x[i]
            value = dx / (
                abs(W[self.equation_system.velocity_index, i]) + 
                self.equation_system.sound_speed(W[:, i]) + self.equation_system.min_var
            )
            max_value = max(max_value, value)
        adaptive_cfl = min(
            self.cfl,
            0.4 if self.reconstruction_method.__name__ == 'weno5' else 0.2 if self.reconstruction_method.__name__ == 'ppm' else self.cfl
        )
        return adaptive_cfl * max_value

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

    def _to_conservative_array(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables for all cells.

        Args:
            W (np.ndarray): Primitive variables, shape (n_vars, n_cells).

        Returns:
            np.ndarray: Conservative variables, shape (n_vars, n_cells).
        """
        if W.ndim != 2 or W.shape[0] != self.equation_system.n_vars:
            raise ValueError(f"W must have shape ({self.equation_system.n_vars}, n_cells)")
        n_cells = W.shape[1]
        U = np.zeros_like(W)
        for i in range(n_cells):
            U[:, i] = self.equation_system.to_conservative(W[:, i])
        return U

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
        if x.size < 3:
            raise ValueError("test must have at least 2 cells (3 points)")
        if U0.ndim != 2 or U0.shape[0] != self.equation_system.n_vars:
            raise ValueError(f"U0 must have shape ({self.equation_system.n_vars}, n_cells)")
        
        min_ghost = 2 if self.reconstruction_method.__name__ in ['ppm', 'weno5'] else 1
        if n_ghost < min_ghost:
            raise ValueError(f"n_ghost must be at least {min_ghost} for {self.reconstruction_method.__name__}")
        
        dx = x[1] - x[0]
        n_cells = U0.shape[1]
        U = U0.copy()
        history = [U.copy()]

        t = 0.0
        n = 1

        while t < T:
            print(f"{n:03d} - {t:.3f} s")

            # Apply boundary conditions
            U_ext = self.bc.apply_bcs(U, n_ghost)

            # Compute time step
            dt = min(self.compute_dt(U, x), T - t)

            # Reconstruct states at interfaces
            UL, UR = self.reconstruction_method(U_ext, dx, n_ghost, use_primitive=self.use_primitive_reconstruction)
            WL = self._to_primitive_array(UL)
            WR = self._to_primitive_array(UR)
            
            # Compute fluxes
            F = np.zeros_like(UL)
            for i in range(UL.shape[1]):
                F[:, i] = self.flux(UL[:, i], UR[:, i], WL[:, i], WR[:, i])

            # Update solution
            dF = (F[:, 1:] - F[:, :-1])
            U = U - (dt / dx) * dF[:, n_ghost - 1:n_cells + n_ghost - 1]

            history.append(U.copy())
            
            t += dt
            n += 1

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
