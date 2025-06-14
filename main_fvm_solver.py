import argparse
import numpy as np

from src.utils import parse_xml_config, create_grid, plot_1d_mesh

from src.boundary import BoundaryCondition

from src.equation.advection_equation import AdvectionEquation
from src.equation.isentropic_gas_equation import IsentropicGas
from src.equation.shallow_water_equation import ShallowWater
from src.equation.euler_equation import EulerEquation

from src.solver import Solver
from src.reconstruction import Reconstruction


def main():
    """
    Run the finite volume solver with XML configuration and save results.

    Reads configuration, sets up the equation system, grid, initial and boundary conditions,
    runs the solver, and plots the final solution snapshot.
    """
    parser = argparse.ArgumentParser(
        description="Finite Volume Riemann Solver for 1D/2D/3D Hyperbolic Conservation Laws"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_shallow_water_dam_break.xml",
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
    # 5. Ensure initial conditions are set with sufficient precision

    # # Example: override some config values for accuracy
    # config['nx'] = max(config['nx'], 400)  # Increase grid points if needed
    # config['reconstruction'] = 'weno'      # Use higher-order if supported
    # config['limiter'] = 'vanleer'          # Use less diffusive limiter
    # config['cfl'] = min(config['cfl'], 0.5)

    # Select equation system
    equation_dict = {
        "euler": EulerEquation(gamma=config["gamma"]),
        "advection": AdvectionEquation(
            advection_speed=float(config.get("advection_speed", 1.0))
        ),
        "isentropic": IsentropicGas(gamma=config["gamma"], k=config["k"]),
        "shallow_water": ShallowWater(gravity=9.81),
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
        equation_system=equation,
        bc_kind=config["bc_type"],
        grid=mesh_grid,
        n_ghost=config["ghost_cells"],
        left_boundary_state=config["left_values"],
        right_boundary_state=config["right_values"],
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
    solver_inst.print_info()

    # Convert to conservative variables
    U0 = equation.to_conservative_batch(W0)

    # Solve and save
    U_history, final_t = solver_inst.solve(U0, config["T"], config["time_integration"])

    print(f"Simulation time: {final_t:.4f} reached")
    if config["equation"] == "advection":
        var_to_show = "u"
    elif config["equation"] == "shallow_water":
        var_to_show = "height"
    else:
        var_to_show = "density"
    solver_inst.plot_solution(U_history, final_t, var_to_show)


if __name__ == "__main__":
    main()
