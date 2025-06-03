import numpy as np
import matplotlib.pyplot as plt
from .equation.base_equation import EquationSystem
from .flux import Flux
from .reconstruction import Reconstruction
from .boundary import BoundaryCondition

class Solver:
    """Godunov-type solver for hyperbolic conservation laws in 1D (extensible to 2D/3D).

    Integrates systems using specified flux, reconstruction, and boundary conditions.
    Supports ASCII output and convergence monitoring.
    """

    def __init__(self, equation_system: EquationSystem, boundary_condition: BoundaryCondition, 
                 grid: np.ndarray,
                 cfl: float = 0.5, flux: str = 'hllc', reconstruction: str = 'weno5', 
                 reconstruct_in_primitive: bool = False, limiter: str = 'minmod',
                 max_iterations: int = 10000, convergence_tol: float = 1e-6,
                 output_filename: str = 'solution.dat'):
        """Initialize the solver for 1D/2D/3D hyperbolic systems.

        Args:
            equation_system (EquationSystem): The equation system to solve.
            boundary_condition (BoundaryCondition): Boundary condition handler.
            grid (np.ndarray): Spatial grid points (1D) or meshgrid (2D/3D).
            cfl (float): CFL number for time step control.
            flux (str): Flux method ('lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe').
            reconstruction (str): Reconstruction method ('piecewise_constant', 'muscl', 'ppm', 'weno5').
            reconstruct_in_primitive (bool): If True, reconstruct in primitive variables; else in conservative.
            limiter (str): Limiter for MUSCL ('minmod', 'superbee', 'vanleer', etc.).
            max_iterations (int): Maximum number of time steps.
            convergence_tol (float): Tolerance for convergence check.
            output_filename (str): File for ASCII solution output.

        Raises:
            TypeError: If equation_system or boundary_condition is invalid.
            ValueError: If flux, reconstruction, cfl, or max_iterations is invalid.
        """
        if not isinstance(equation_system, EquationSystem):
            raise TypeError("equation_system must be an EquationSystem instance")
        if not isinstance(boundary_condition, BoundaryCondition):
            raise TypeError("boundary_condition must be a BoundaryCondition instance")
        if cfl <= 0:
            raise ValueError("cfl must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if convergence_tol <= 0:
            raise ValueError("convergence_tolerance must be positive")

        self.equation_system = equation_system
        self.bc = boundary_condition
        self.grid = grid
        self.flux = Flux(equation_system, lambda_max=1.0).get_flux(flux.lower())
        self.reconstruction = Reconstruction(equation_system, reconstruct_in_primitive, limiter=limiter if reconstruction.lower() == 'muscl' else "")
        self.reconstruction_method = {
            'piecewise_constant': self.reconstruction.piecewise_constant,
            'muscl': self.reconstruction.muscl,
            # 'ppm': self.reconstruction.ppm,
            # 'weno5': self.reconstruction.weno5
        }[reconstruction.lower()]
        self.cfl = cfl
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.output_filename = output_filename
        self.variable_names = equation_system.var_names

    def compute_dt(self, U: np.ndarray) -> float:
        """Compute time step based on CFL condition for 1D/2D/3D grids.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells) or higher for 2D/3D.

        Returns:
            float: Time step size.

        Raises:
            ValueError: If grid or U shape is invalid.
        """
        grid = self.grid
        if grid.ndim == 1:
            if grid.size != U.shape[1] + 1:
                raise ValueError("grid must be of length n_cells + 1")
        if U.ndim < 2 or U.shape[0] != self.equation_system.num_vars:
            raise ValueError(f"U must have shape ({self.equation_system.num_vars}, n_cells, ...)")

        W = self.equation_system.to_primitive_batch(U)
        min_value = float('inf')
        if U.ndim == 2:  # 1D
            n_cells = U.shape[1]
            for i in range(n_cells):
                dx = grid[i+1] - grid[i]
                value = dx / max(abs(W[self.equation_system.vel_idx, i]) + self.equation_system.sound_speed(W[:, i]),
                                self.equation_system.min_value)
                min_value = min(min_value, value)
        else:  # 2D/3D
            dx = grid[0][1] - grid[0][0]
            for idx in np.ndindex(U.shape[1:]):
                value = dx / max(abs(W[self.equation_system.vel_idx, idx]) + self.equation_system.sound_speed(W[:, idx]),
                                self.equation_system.min_value)
                min_value = min(min_value, value)
        adaptive_cfl = min(
            self.cfl,
            0.4 if self.reconstruction_method.__name__ == 'weno5' else 0.2 if self.reconstruction_method.__name__ == 'ppm' else self.cfl
        )
        return adaptive_cfl * min_value

    def save_solution(self, U: np.ndarray, t: float, step: int):
        """Save solution to ASCII file for each time step.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells, ...).
            t (float): Current time.
            step (int): Time step index.
        """
        W = self.equation_system.to_primitive_batch(U)
        grid = self.grid
        with open(self.output_filename, 'a') as f:
            f.write(f"# Step {step}, Time {t:.6f}\n")
            if U.ndim == 2:  # 1D
                xmid = (grid[1:] + grid[:-1]) / 2
                f.write("# x " + " ".join(self.variable_names) + "\n")
                for i in range(U.shape[1]):
                    f.write(f"{xmid[i]:.6e} " + " ".join(f"{w:.6e}" for w in W[:, i]) + "\n")
            else:  # 2D/3D
                f.write("# Structured grid output\n")
                for idx in np.ndindex(U.shape[1:]):
                    f.write(f"{' '.join(str(i) for i in idx)} " + " ".join(f"{w:.6e}" for w in W[:, idx]) + "\n")
            f.write("\n")

    def solve(self, U0: np.ndarray, T: float, n_ghost: int = 2) -> tuple:
        """
        Solve the hyperbolic system until time T in 1D/2D/3D.

        Uses Godunov-type method with specified flux, reconstruction, and boundary conditions.
        Saves solution to ASCII file and monitors convergence.

        Args:
            U0 (np.ndarray): Initial conservative variables, shape (n_vars, n_cells, ...).
            T (float): Final simulation time.
            n_ghost (int): Number of ghost cells per side.

        Returns:
            tuple: (history, t), where history is array of states [n_steps, n_vars, n_cells, ...], t is final time.

        Raises:
            ValueError: If inputs are invalid or n_ghost is insufficient.
            RuntimeError: If solution does not converge within max_iterations or tolerance.
        """
        grid = self.grid
        if T < 0:
            raise ValueError("T must be non-negative")
        if grid.ndim == 1 and grid.size < 3:
            raise ValueError("Grid must have at least 2 cells (3 points)")
        if U0.ndim < 2 or U0.shape[0] != self.equation_system.num_vars:
            raise ValueError(f"U0 must have shape ({self.equation_system.num_vars}, n_cells, ...)")
        
        min_ghost = 2 if self.reconstruction_method.__name__ in ['ppm', 'weno5'] else 1
        if n_ghost < min_ghost:
            raise ValueError(f"n_ghost must be at least {min_ghost} for {self.reconstruction_method.__name__}")

        # Clear output file
        open(self.output_filename, 'w').close()

        n_cells = U0.shape[1]
        n_cells_total = n_cells + 2 * n_ghost

        grid_dx = self.grid[1:] - self.grid[0:-1]
        grid_dx_expand = np.zeros(n_cells_total)
        grid_dx_expand[n_ghost:n_ghost + n_cells] = grid_dx
        grid_dx_expand[:n_ghost] = grid_dx[0]
        grid_dx_expand[n_cells + n_ghost:] = grid_dx[-1]

        U = U0.copy()
        history = [U.copy()]
        t = 0.0
        n = 0
        prev_norm = float('inf')

        while t < T and n < self.max_iterations:
            # Apply boundary conditions, expanding grid
            U_ext = self.bc.apply_bcs(U, n_ghost)

            # Compute time step
            dt = min(self.compute_dt(U), T - t)

            # Reconstruct states at interfaces
            UL, UR = self.reconstruction_method(U_ext, grid_dx_expand, n_ghost)
            
            # Compute fluxes
            if U.ndim == 2:  # 1D
                F = np.zeros_like(UL)
                WL = self.equation_system.to_primitive_batch(UL)
                WR = self.equation_system.to_primitive_batch(UR)
                for i in range(UL.shape[1]):
                    F[:, i] = self.flux(UL[:, i], UR[:, i], WL[:, i], WR[:, i])
                
                dF = F[:, 1:] - F[:, :-1]
                U_new = U - (dt / grid_dx) * dF[:, n_ghost-1 : n_cells + n_ghost - 1]

            else:  # 2D/3D
                U_new = np.zeros_like(U)
                raise NotImplementedError("2D/3D flux computation not yet implemented")

            # Check convergence
            normU = np.linalg.norm(U)
            residual = np.linalg.norm(U_new - U) / normU if normU > 0 else np.linalg.norm(U_new)

            U = U_new
            self.save_solution(U, t, n)
            history.append(U.copy())
            
            t += dt
            n += 1

            # Print progress
            percent = (t / T * 100) if T != 0 else 0.0
            print(f"\rStep {n:03d} | Time: {t:.4f} / {T:.3f} ({percent:5.1f}%) | Δt = {dt:.4f} s | Residual {residual * 100:5.2f} %", end='\n', flush=True)

            # Check convergence tolerance
            if residual < self.convergence_tol:
                print(f"Converged at step {n}, residual {residual:.6e}")
                break

            if n >= self.max_iterations:
                raise RuntimeError("Maximum iterations reached without convergence")

        print(f"Simulation completed: {n} steps, final time {t:.6f}")
        return np.array(history), t

    def plot_solution(self, history: np.ndarray, t: float, variable: str = ""):
        """Plot the solution at the final time for 1D (2D/3D plotting not implemented).

        Only the last snapshot is shown.

        Args:
            history (np.ndarray): Solution history, shape (n_steps, n_vars, n_cells, ...).
            t (float): Final time.
            variable (str): Variable to plot (optional, default: all variables).

        Raises:
            ValueError: If variable is invalid or history shape is incorrect.
        """
        grid = self.grid
        x = grid
        if history.ndim < 3 or history.shape[1] != self.equation_system.num_vars:
            raise ValueError(f"history must have shape (n_steps, {self.equation_system.num_vars}, n_cells, ...)")
        
        U_final = history[-1]
        W_final = self.equation_system.to_primitive_batch(U_final)
        
        if x.ndim == 1:  # 1D
            xmid = (x[1:] + x[:-1]) / 2
            plt.figure()
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
        else:
            raise NotImplementedError("2D/3D plotting not yet implemented")
