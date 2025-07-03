import numpy as np
import matplotlib.pyplot as plt
from .equation.equation_base import EqnBase
from .boundary import BoundaryCondition
from .reconstruction import Reconstruction


class Solver:
    """Godunov-type solver for hyperbolic conservation laws in 1D (extensible to 2D/3D).

    Integrates systems using specified flux, reconstruction, and boundary conditions.
    Supports ASCII output and convergence monitoring.
    """

    def __init__(
        self,
        eqn_obj: EqnBase,
        bc_obj: BoundaryCondition,
        mesh_obj: np.ndarray,
        reconst_obj: Reconstruction,
        n_ghost: int = 2,
        cfl: float = 0.5,
        max_iterations: int = 10000,
        convergence_tol: float = 1e-6,
        output_filename: str = "solution.dat",
    ):
        """Initialize the solver for 1D/2D/3D hyperbolic systems.

        Args:
            equation (EquationBase): The equation system to solve.
            bc_obj (BoundaryCondition): Boundary condition handler.
            grid (np.ndarray): Spatial grid points (1D) or meshgrid (2D/3D).
            cfl (float): CFL number for time step control.
            max_iterations (int): Maximum number of time steps.
            convergence_tol (float): Tolerance for convergence check.
            output_filename (str): File for ASCII solution output.

        Raises:
            TypeError: If equation or bc_obj is invalid.
            ValueError: If flux, reconstruction, cfl, or max_iterations is invalid.
        """
        if not isinstance(eqn_obj, EqnBase):
            raise TypeError("equation must be an EquationBase instance")
        if not isinstance(bc_obj, BoundaryCondition):
            raise TypeError("bc_obj must be a BoundaryCondition instance")
        if not isinstance(mesh_obj, np.ndarray):
            raise TypeError("mesh_obj must be a ndarray")
        if not isinstance(reconst_obj, Reconstruction):
            raise TypeError("reconst_obj must be a Reconstruction instance")

        if cfl > 1 or cfl <= 0:
            raise ValueError("cfl must be within [0, 1]")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if convergence_tol <= 0:
            raise ValueError("convergence_tolerance must be positive")

        self.bc_obj = bc_obj
        self.mesh_obj = mesh_obj
        self.equation = eqn_obj
        self.reconst_obj = reconst_obj
        self.cfl = cfl
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.output_filename = output_filename

        self.num_vars = self.equation.num_vars
        self.variable_names = self.equation.var_names

        n_cells = self.mesh_obj.size - 1

        self.n_ghost = n_ghost
        self.n_cells = n_cells
        self.n_cells_total = n_cells + 2 * n_ghost

        dx_origin = self.mesh_obj[1:] - self.mesh_obj[0:-1]
        self.dx = dx_origin

        dx_aug = np.zeros(self.n_cells_total)
        dx_aug[n_ghost : n_ghost + n_cells] = dx_origin
        dx_aug[:n_ghost] = dx_origin[0]
        dx_aug[n_cells + n_ghost :] = dx_origin[-1]
        self.dx_aug = dx_aug

    def print_info(self):
        """Print useful information about the solver configuration and setup."""

        print("========================================")
        print("Solver Information:")
        print(f"  Equation system: {type(self.equation).__name__}")
        print(
            f"  Number of variables: {self.num_vars} ({', '.join(self.variable_names)})"
        )
        print(f"  Mesh points: {self.mesh_obj.size}")
        print(f"  Number of cells: {self.n_cells}")
        print(f"  Number of ghost cells: {self.n_ghost}")
        print(
            f"  Reconstruction: {type(self.reconst_obj).__name__} ({getattr(self.reconst_obj, 'name', 'unknown')})"
        )
        print(
            f"  Flux: {type(self.reconst_obj.flux_obj).__name__} ({getattr(self.reconst_obj.flux_obj, 'name', 'unknown')})"
        )
        print(
            f"  Limiter: {type(self.reconst_obj.limiter_obj).__name__} ({getattr(self.reconst_obj.limiter_obj, 'name', 'unknown')})"
        )
        print(f"  CFL number: {self.cfl}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Convergence tolerance: {self.convergence_tol}")
        print(f"  Output file: {self.output_filename}")
        print(
            f"  dx (cell widths): min={np.min(self.dx):.3e}, max={np.max(self.dx):.3e}, mean={np.mean(self.dx):.3e}"
        )
        print("========================================")

    # function to augment the grid with ghost cells
    def augment_vars(self, U: np.ndarray) -> np.ndarray:
        """Augment conservative variables with ghost cells.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells).

        Returns:
            np.ndarray: Augmented conservative variables with ghost cells.
        """
        if U.ndim != 2 or U.shape[0] != self.num_vars or U.shape[1] != self.n_cells:
            raise ValueError(f"U must have shape ({self.num_vars}, {self.n_cells})")

        U_aug = np.zeros((self.num_vars, self.n_cells_total))
        U_aug[:, self.n_ghost : self.n_ghost + self.n_cells] = U
        return U_aug

    def de_augment_vars(self, U_aug: np.ndarray) -> np.ndarray:
        """Remove ghost cells from augmented conservative variables.

        Args:
            U_aug (np.ndarray): Augmented conservative variables, shape (n_vars, n_cells_total).

        Returns:
            np.ndarray: Conservative variables without ghost cells.
        """
        if (
            U_aug.ndim != 2
            or U_aug.shape[0] != self.num_vars
            or U_aug.shape[1] != self.n_cells_total
        ):
            raise ValueError(
                f"U_aug must have shape ({self.num_vars}, {self.n_cells_total})"
            )

        return U_aug[:, self.n_ghost : self.n_ghost + self.n_cells]

    def compute_dt(self, U_aug: np.ndarray) -> float:
        """Compute time step based on CFL condition for 1D/2D/3D grids.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells) or higher for 2D/3D.

        Returns:
            float: Time step size.

        Raises:
            ValueError: If grid or U shape is invalid.
        """
        if (
            U_aug.ndim < 2
            or U_aug.shape[0] != self.num_vars
            or U_aug.shape[1] != self.n_cells_total
        ):
            raise ValueError(
                f"U must have shape ({self.num_vars}, {self.n_cells_total})"
            )

        dx = self.dx_aug

        W_aug = self.equation.to_primitive_batch(U_aug)
        min_value = float("inf")

        for i in range(self.n_ghost, self.n_cells + self.n_ghost):
            value = dx[i] / max(
                abs(self.equation.max_eigenvalue(U_aug[:, i])), self.equation.min_value
            )
            min_value = min(min_value, value)

        adaptive_cfl = min(
            self.cfl,
            (
                0.4
                if self.reconst_obj.name == "weno5"
                else 0.2 if self.reconst_obj.name == "ppm" else self.cfl
            ),
        )
        return adaptive_cfl * min_value

    def save_solution(self, U: np.ndarray, t: float, step: int):
        """Save solution to ASCII file for each time step.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells, ...).
            t (float): Current time.
            step (int): Time step index.
        """
        W = self.equation.to_primitive_batch(U)
        grid = self.mesh_obj
        with open(self.output_filename, "a") as f:
            f.write(f"# Step {step}, Time {t:.6f}\n")
            if U.ndim == 2:  # 1D
                xmid = (grid[1:] + grid[:-1]) / 2
                f.write("# x " + " ".join(self.variable_names) + "\n")
                for i in range(U.shape[1]):
                    f.write(
                        f"{xmid[i]:.6e} " + " ".join(f"{w:.6e}" for w in W[:, i]) + "\n"
                    )
            else:  # 2D/3D
                f.write("# Structured grid output\n")
                for idx in np.ndindex(U.shape[1:]):
                    f.write(
                        f"{' '.join(str(i) for i in idx)} "
                        + " ".join(f"{w:.6e}" for w in W[:, idx])
                        + "\n"
                    )
            f.write("\n")

    def solve(self, U0: np.ndarray, T: float, time_integration_method: str) -> tuple:
        """
        Solve the hyperbolic system until time T in 1D/2D/3D.

        Uses Godunov-type method with specified flux, reconstruction, and boundary conditions.
        Saves solution to ASCII file and monitors convergence.

        Args:
            U0 (np.ndarray): Initial conservative variables, shape (n_vars, n_cells + 2 * n_ghost, ...).
            T (float): Final simulation time.

        Returns:
            tuple: (history, t), where history is array of states [n_steps, n_vars, n_cells, ...], t is final time.

        Raises:
            ValueError: If inputs are invalid or n_ghost is insufficient.
            RuntimeError: If solution does not converge within max_iterations or tolerance.
        """
        if T < 0:
            raise ValueError("T must be non-negative")
        if self.mesh_obj.ndim == 1 and self.mesh_obj.size < 3:
            raise ValueError("Grid must have at least 2 cells (3 points)")
        if U0.ndim < 2 or U0.shape[0] != self.num_vars or U0.shape[1] != self.n_cells:
            raise ValueError(
                f"U0 must be argumented and have shape ({self.num_vars}, {self.n_cells_total}, ...)"
            )

        min_ghost = 2 if self.reconst_obj.name in ["ppm", "weno5"] else 1
        if self.n_ghost < min_ghost:
            raise ValueError(
                f"n_ghost must be at least {min_ghost} for {self.reconst_obj.name}"
            )

        U0_aug = self.augment_vars(U0)

        # Clear output file
        open(self.output_filename, "w").close()

        # Enforce boundary conditions on initial state
        self.bc_obj.enforce_bc(U0_aug)
        U = U0_aug.copy()

        history = [U0.copy()]
        t = 0.0
        n = 0
        prev_residual = float("inf")

        while t < T and n < self.max_iterations:

            # Compute time step
            dt = min(self.compute_dt(U), T - t)
            t += dt
            n += 1

            # chosen between RK2 and 1st order update
            # RK2 method
            if time_integration_method == "rk2":
                # RK2 1st step
                U_star = U - dt * self.reconst_obj.reconst_func(U, self.dx_aug)
                self.bc_obj.enforce_bc(U_star)

                # RK2 2nd step / update U
                U_new = (
                    U + U_star - dt * self.reconst_obj.reconst_func(U_star, self.dx_aug)
                ) / 2.0
                self.bc_obj.enforce_bc(U_new)

            elif time_integration_method == "euler":  # 1st order update
                # 1st order update
                U_new = U - dt * self.reconst_obj.reconst_func(U, self.dx_aug)
                self.bc_obj.enforce_bc(U_new)

            # raise error if not the above
            else:
                raise RuntimeError(
                    f"time integration_method method {time_integration_method} not implemented yet"
                )

            # Check residual for possible convergence
            normU = np.linalg.norm(U)
            residual = (
                np.linalg.norm(U_new - U) / normU
                if normU > 0
                else np.linalg.norm(U_new)
            )

            U = U_new

            U_save = self.de_augment_vars(U)
            self.save_solution(U_save, t, n)
            history.append(U_save)

            # Print progress
            percent = (t / T * 100) if T != 0 else 0.0
            print(
                f"\rStep {n:03d} | Δt = {dt:.4f} s | Residual {residual * 100:5.2f} % | Time: {t:.3f} / {T:.2f} s ({percent:4.1f}%)",
                end="\n",
                flush=True,
            )

            # check if residual is suddenly increasing by 5 times
            if n > 0 and residual > 5 * prev_residual:
                raise RuntimeError(
                    f"Possible divergence; residual increased by more than 5 times: {residual / prev_residual:.2g}"
                )
            prev_residual = residual

            # Check convergence tolerance
            if residual < self.convergence_tol:
                print(f"Converged at step {n}, residual {residual:.6e}")
                break

        print(f"\nSimulation completed after {n} steps, final time {t:.6f} s")
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
        x = self.mesh_obj
        if history.ndim < 3 or history.shape[1] != self.num_vars:
            raise ValueError(
                f"history must have shape (n_steps, {self.num_vars}, n_cells, ...)"
            )

        U_final = history[-1]
        W_final = self.equation.to_primitive_batch(U_final)

        if x.ndim == 1:  # 1D
            xmid = (x[1:] + x[:-1]) / 2
            plt.figure()
            if variable:
                if variable not in self.variable_names:
                    raise ValueError(
                        f"Variable {variable} not in {self.variable_names}"
                    )
                idx = self.variable_names.index(variable)
                plt.plot(xmid, W_final[idx, :], label=variable)
                plt.xlabel("x")
                plt.ylabel(variable)
                plt.title(f"{variable} at t = {t:.3f}")
            else:
                for i, name in enumerate(self.variable_names):
                    plt.plot(xmid, W_final[i, :], label=name)
                plt.xlabel("x")
                plt.ylabel("Variables")
                plt.title(f"Solution at t = {t:.3f}")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            raise NotImplementedError("2D/3D plotting not yet implemented")
