"""
Main execution script for the 1D Finite Volume Method (FVM) solver.

This script orchestrates the entire simulation process. It parses command-line
arguments to get the configuration file, sets up the simulation environment
(equation, grid, boundary conditions, initial conditions), initializes the
solver, runs the simulation, and plots the final results.
"""

import argparse
import numpy as np

# Utility functions for configuration and grid creation
from src.utils import parse_xml_config, create_grid

# Import all available equation models
from src.equation.equation_advection import EqnAdvection
from src.equation.equation_burgers import EqnBurgers
from src.equation.equation_euler import EqnEuler
from src.equation.equation_isentropic_gas import EqnIsentropicGas
from src.equation.equation_keyfitz_kranzer import EqnKK
from src.equation.equation_shallow_water import EqnShallowWater
from src.equation.equation_traffic_flow import EqnTrafficLWR

# Core solver components
from src.solver import Solver
from src.boundary import BoundaryCondition
from src.reconstruction import Reconstruction


def main():
    """
    Main function to set up and run the FVM simulation.

    This function performs the following steps:
    1. Parses the XML configuration file specified via command-line arguments.
    2. Selects and initializes the appropriate equation system.
    3. Creates the computational grid.
    4. Sets up the boundary and initial conditions.
    5. Configures the reconstruction and numerical flux methods.
    6. Initializes and runs the main solver.
    7. Plots the final solution and prints a summary of the setup.
    """
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="A Godunov-type Riemann Solver for 1D Hyperbolic Conservation Laws"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/shallow_water_dam_break.xml",
        help="Path to the XML configuration file for the simulation.",
    )
    args = parser.parse_args()

    # --- Configuration Setup ---
    config = parse_xml_config(args.config)

    # --- Equation System Selection ---
    equation_dict = {
        "advection": EqnAdvection(speed=float(config.get("speed", 1.0))),
        "burgers": EqnBurgers(),
        "euler": EqnEuler(gamma=config["gamma"]),
        "isentropic": EqnIsentropicGas(gamma=config["gamma"], k=config["k"]),
        "keyfitz_kranzer": EqnKK(),
        "shallow_water": EqnShallowWater(gravity=9.81),
        "traffic_flow": EqnTrafficLWR(rho_max=config["rhoMax"], v_max=config["vMax"]),
    }
    equation = equation_dict.get(config["equation"].lower())
    if equation is None:
        raise ValueError(f"Equation type '{config['equation']}' is not supported.")

    # --- Grid Generation ---
    mesh_grid = create_grid(
        config["xmin"], config["xmax"], config["nx"], config.get("stretch_factor", 1.0)
    )
    # plot_1d_mesh(mesh_grid)

    # --- Boundary Conditions ---
    bc_inst = BoundaryCondition(
        eqn_obj=equation,
        bc_kind=config["bc"]["type"],
        grid=mesh_grid,
        n_ghost=config["ghost_cells"],
        left_boundary_state=config["bc"]["left"],
        right_boundary_state=config["bc"]["right"],
    )

    # --- Spatial Reconstruction ---
    reconst_inst = Reconstruction(
        eqn_obj=equation,
        str_reconst=config["reconstruction"],
        str_flux=config["flux"],
        str_limiter=config["limiter"],
        str_domain=config["reconstruction_vars"],
    )

    # --- Initial Conditions ---
    n_vars = equation.num_vars
    W0 = np.zeros((n_vars, config["nx"]), dtype=float)
    # Determine the split point for the initial Riemann problem
    split_idx = np.argmin(np.abs(mesh_grid - config["ic"]["split"]))
    W0[:, :split_idx] = np.array(config["ic"]["left"])[:, np.newaxis]
    W0[:, split_idx:] = np.array(config["ic"]["right"])[:, np.newaxis]
    # Convert initial primitive variables to conservative variables
    U0 = equation.to_conservative_batch(W0)

    # --- Solver Initialization ---
    solver_inst = Solver(
        eqn_obj=equation,
        bc_obj=bc_inst,
        mesh_obj=mesh_grid,
        reconst_obj=reconst_inst,
        n_ghost=config["ghost_cells"],
        cfl=config["cfl"],
        max_iterations=config.get("max_iterations", 10000),
        convergence_tol=config.get("convergence_tolerance", 1e-6),
        output_filename=config["output_filename"],
    )

    # --- Run Simulation ---
    solver_inst.print_info()
    U_history, final_t = solver_inst.solve(U0, config["T"], config["time_integration"])

    # --- Plotting and Finalization ---
    # By default, plot the first variable in the system
    var_to_show = equation.var_names[0]
    solver_inst.plot_solution(U_history, final_t, var_to_show)


if __name__ == "__main__":
    main()
