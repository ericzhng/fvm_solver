import argparse
import numpy as np

from src.utils import parse_xml_config, create_grid

from src.equation.equation_advection import EqnAdvection
from src.equation.equation_burgers import EqnBurgers
from src.equation.equation_euler import EqnEuler
from src.equation.equation_isentropic_gas import EqnIsentropicGas
from src.equation.equation_keyfitz_kranzer import EqnKK
from src.equation.equation_shallow_water import EqnShallowWater
from src.equation.equation_traffic_flow import EqnTrafficLWR

from src.solver import Solver
from src.boundary import BoundaryCondition
from src.reconstruction import Reconstruction


def main():
    """
    Run the finite volume solver with XML configuration and save results.

    Reads configuration, sets up the equation system, grid, initial and boundary conditions,
    runs the solver, and plots the final solution snapshot.
    """
    parser = argparse.ArgumentParser(
        description="Riemann Solver for 1D Hyperbolic Conservation Laws"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/keyfitz_kranzer_step.xml",
        help="Path to XML configuration file",
    )
    args = parser.parse_args()

    # Parse XML configuration
    config = parse_xml_config(args.config)

    # --- Accuracy improvements ---
    # 1. Use more grid points for better resolution (if not already high)
    # 2. Use a higher-order reconstruction (e.g., 'weno', 'ppm', or 'mp5' if available)
    # 3. Use a less diffusive limiter (e.g., 'vanleer', 'mc', or 'superbee')
    # 4. Reduce CFL number (e.g., 0.5 or lower) for stability and accuracy

    # Select equation system
    equation_dict = {
        "advection": EqnAdvection(speed=float(config.get("speed", 1.0))),
        "burgers": EqnBurgers(),
        "euler": EqnEuler(gamma=config["gamma"]),
        "isentropic": EqnIsentropicGas(gamma=config["gamma"], k=config["k"]),
        "keyfitz_kranzer": EqnKK(),
        "shallow_water": EqnShallowWater(gravity=9.81),
        "traffic_flow": EqnTrafficLWR(rhoMax=config["rhoMax"], vMax=config["vMax"]),
    }
    equation = equation_dict[config["equation"]]

    # Set up grid
    mesh_grid = create_grid(
        config["xmin"],
        config["xmax"],
        config["nx"],
        config["stretch_factor"],
    )
    # plot_1d_mesh(mesh_grid)

    # Set up boundary conditions
    bc_inst = BoundaryCondition(
        eqn_obj=equation,
        bc_kind=config["bc"]["type"],
        grid=mesh_grid,
        n_ghost=config["ghost_cells"],
        left_boundary_state=config["bc"]["left"],
        right_boundary_state=config["bc"]["right"],
    )

    # Set up reconstruction instance
    reconst_inst = Reconstruction(
        eqn_obj=equation,
        str_reconst=config["reconstruction"],
        str_flux=config["flux"],
        str_limiter=config["limiter"],
        str_domain=config["reconstruction_vars"],
    )

    # Set initial conditions
    n_vars = len(equation.get_var_names())
    W0 = np.zeros((n_vars, config["nx"]), dtype=float)

    # Find the index where mesh_grid is closest to x=0.5
    split_idx = np.argmin(np.abs(mesh_grid - config["ic"]["split"]))
    W0[:, :split_idx] = np.array(config["ic"]["left"], dtype=float)[:, np.newaxis]
    W0[:, split_idx:] = np.array(config["ic"]["right"], dtype=float)[:, np.newaxis]

    # Convert to conservative variables
    U0 = equation.to_conservative_batch(W0)

    # Initialize solver
    solver_inst = Solver(
        eqn_obj=equation,
        bc_obj=bc_inst,
        mesh_obj=mesh_grid,
        reconst_obj=reconst_inst,
        n_ghost=config["ghost_cells"],
        cfl=config["cfl"],
        max_iterations=config["max_iterations"],
        convergence_tol=config["convergence_tolerance"],
        output_filename=config["output_filename"],
    )
    # Solve and save
    U_history, final_t = solver_inst.solve(U0, config["T"], config["time_integration"])

    var_to_show = equation.var_names[0]
    solver_inst.plot_solution(U_history, final_t, var_to_show)

    solver_inst.print_info()


if __name__ == "__main__":
    main()
