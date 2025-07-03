"""
This module defines the main Solver class for the 1D finite volume framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from .equation.equation_base import EqnBase
from .boundary import BoundaryCondition
from .reconstruction import Reconstruction


class Solver:
    """
    A Godunov-type finite volume solver for 1D hyperbolic conservation laws.

    This class orchestrates the numerical solution of a given hyperbolic system.
    It integrates the solution in time using a specified time integration scheme
    (e.g., Euler forward, RK2) and manages the core steps of a finite volume
    method: boundary condition enforcement, spatial reconstruction, and numerical
    flux calculation.

    Attributes:
        bc_obj (BoundaryCondition): The boundary condition handler.
        mesh_obj (np.ndarray): The 1D spatial grid.
        equation (EqnBase): The equation system being solved.
        reconst_obj (Reconstruction): The spatial reconstruction handler.
        cfl (float): The Courant-Friedrichs-Lewy (CFL) number for stability.
        max_iterations (int): The maximum number of time steps to perform.
        convergence_tol (float): The tolerance for the solution residual to
                                 determine convergence.
        output_filename (str): The name of the file for solution output.
        num_vars (int): The number of variables in the system.
        variable_names (list[str]): The names of the primitive variables.
        n_ghost (int): The number of ghost cells on each side of the domain.
        n_cells (int): The number of computational cells.
        n_cells_total (int): The total number of cells including ghost cells.
        dx (np.ndarray): Array of cell widths for the computational domain.
        dx_aug (np.ndarray): Array of cell widths for the augmented domain.
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
        """
        Initializes the finite volume solver.

        Args:
            eqn_obj (EqnBase): An instance of an equation class.
            bc_obj (BoundaryCondition): An instance of the boundary condition handler.
            mesh_obj (np.ndarray): A 1D array representing the spatial grid.
            reconst_obj (Reconstruction): An instance of the reconstruction handler.
            n_ghost (int, optional): The number of ghost cells. Defaults to 2.
            cfl (float, optional): The CFL number for time step calculation.
                                   Must be between 0 and 1. Defaults to 0.5.
            max_iterations (int, optional): Maximum number of iterations.
                                          Defaults to 10000.
            convergence_tol (float, optional): Residual tolerance for convergence.
                                               Defaults to 1e-6.
            output_filename (str, optional): File to save the solution history.
                                             Defaults to "solution.dat".
        """
        # Validate input types
        if not isinstance(eqn_obj, EqnBase):
            raise TypeError("eqn_obj must be an instance of EqnBase.")
        if not isinstance(bc_obj, BoundaryCondition):
            raise TypeError("bc_obj must be a BoundaryCondition instance.")
        if not isinstance(mesh_obj, np.ndarray):
            raise TypeError("mesh_obj must be a NumPy array.")
        if not isinstance(reconst_obj, Reconstruction):
            raise TypeError("reconst_obj must be a Reconstruction instance.")

        # Validate input values
        if not (0 < cfl <= 1):
            raise ValueError("CFL number must be in the range (0, 1].")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

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
        """Prints a summary of the solver's configuration."""
        print("========================================")
        print("      Finite Volume Solver Setup      ")
        print("========================================")
        print(f"  Equation System:    {type(self.equation).__name__}")
        print(f"  Variables:          {', '.join(self.variable_names)}")
        print(f"  Grid Cells:         {self.n_cells}")
        print(f"  Ghost Cells:        {self.n_ghost}")
        print(f"  Reconstruction:     {self.reconst_obj.name}")
        print(f"  Numerical Flux:     {self.reconst_obj.flux_obj.name}")
        if self.reconst_obj.limiter_obj:
            print(f"  Slope Limiter:      {self.reconst_obj.limiter_obj.name}")
        print(f"  CFL Number:         {self.cfl}")
        print(f"  Max Iterations:     {self.max_iterations}")
        print(f"  Convergence Tol:    {self.convergence_tol}")
        print(f"  Output File:        {self.output_filename}")
        print("========================================")

    def augment_vars(self, U: np.ndarray) -> np.ndarray:
        """
        Adds ghost cell placeholders to a conservative variable array.

        Args:
            U (np.ndarray): The conservative variable array for the main domain.
                            Shape: (num_vars, n_cells).

        Returns:
            np.ndarray: An augmented array with space for ghost cells.
                        Shape: (num_vars, n_cells + 2 * n_ghost).
        """
        if U.shape != (self.num_vars, self.n_cells):
            raise ValueError(
                f"Input U must have shape ({self.num_vars}, {self.n_cells})"
            )
        U_aug = np.zeros((self.num_vars, self.n_cells_total))
        U_aug[:, self.n_ghost : self.n_ghost + self.n_cells] = U
        return U_aug

    def de_augment_vars(self, U_aug: np.ndarray) -> np.ndarray:
        """
        Removes ghost cells from an augmented conservative variable array.

        Args:
            U_aug (np.ndarray): The augmented array with ghost cells.

        Returns:
            np.ndarray: The array for the main computational domain.
        """
        return U_aug[:, self.n_ghost : -self.n_ghost]

    def compute_dt(self, U_aug: np.ndarray) -> float:
        """
        Computes the time step size (dt) based on the CFL condition.

        dt = CFL * min(dx_i / |λ_max|_i)

        Args:
            U_aug (np.ndarray): The augmented conservative variable array.

        Returns:
            float: The stable time step size.
        """
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
        """
        Saves the current solution state to the output file.

        Args:
            U (np.ndarray): The conservative variables for the main domain.
            t (float): The current simulation time.
            step (int): The current iteration number.
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

    def solve(
        self, U0: np.ndarray, T: float, time_integration_method: str
    ) -> tuple[np.ndarray, float]:
        """
        The main solver loop.

        Integrates the solution from t=0 to t=T using the specified method.

        Args:
            U0 (np.ndarray): The initial condition for the conservative variables.
                             Shape: (num_vars, n_cells).
            T (float): The final simulation time.
            time_integration_method (str): The time integration scheme to use.
                                           Supported: 'euler', 'rk2'.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the solution history
                                      and the final simulation time.
        """
        if T < 0:
            raise ValueError("Final time T must be non-negative.")
        if U0.shape != (self.num_vars, self.n_cells):
            raise ValueError(
                f"Initial condition U0 must have shape ({self.num_vars}, {self.n_cells})"
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
                raise NotImplementedError(
                    f"Time integration method '{time_integration_method}' is not supported."
                )

            # --- Convergence & Output --- #
            norm_U = np.linalg.norm(self.de_augment_vars(U))
            residual = np.linalg.norm(self.de_augment_vars(U_new - U)) / (norm_U + 1e-9)

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
                print("\nConvergence criteria met.")
                break

        print(f"\nSimulation finished at t={t:.4f}s after {n} iterations.")
        return np.array(history), t

    def plot_solution(self, history: np.ndarray, t: float, variable: str = ""):
        """
        Plots the solution at the final time step.

        Args:
            history (np.ndarray): The solution history from the solve method.
            t (float): The final simulation time.
            variable (str, optional): The name of a specific variable to plot.
                                      If empty, all variables are plotted.
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
