import argparse
import numpy as np

from src.utils import parse_xml_config, gen_grid  # Make sure this import path matches your project structure

from src.boundary import BoundaryCondition
from src.equation.isentropic_gas_equation import IsentropicGas
from src.equation.shallow_water_equation import ShallowWater
from src.equation.euler_equation import EulerEquation
from src.equation.advection_equation import AdvectionEquation
from src.solver import Solver

def main():
    """
    Run the finite volume solver with XML configuration and save results.

    Reads configuration, sets up the equation system, grid, initial and boundary conditions,
    runs the solver, and plots the final solution snapshot.
    """
    parser = argparse.ArgumentParser(description='Finite Volume Riemann Solver for 1D/2D/3D Hyperbolic Conservation Laws')
    parser.add_argument('--config', type=str, default='config_euler_sod_shock_tube.xml', help='Path to XML configuration file')
    args = parser.parse_args()

    config = parse_xml_config(args.config)

    # --- Accuracy improvements ---
    # 1. Use more grid points for better resolution (if not already high)
    # 2. Use a higher-order reconstruction (e.g., 'weno', 'ppm', or 'mp5' if available)
    # 3. Use a less diffusive limiter (e.g., 'vanleer', 'mc', or 'superbee')
    # 4. Reduce CFL number (e.g., 0.5 or lower) for stability and accuracy
    # 5. Ensure initial conditions are set with sufficient precision

    # Example: override some config values for accuracy
    # config['nx'] = max(config['nx'], 400)  # Increase grid points if needed
    # config['reconstruction'] = 'weno'      # Use higher-order if supported
    # config['limiter'] = 'vanleer'          # Use less diffusive limiter
    # config['cfl'] = min(config['cfl'], 0.5)

    # Select equation system
    equation_systems = {
        'euler': EulerEquation(gamma=config['gamma']),
        'isentropic': IsentropicGas(gamma=config['gamma'], k=config['k']),
        'shallow_water': ShallowWater(gravity=9.81),
        'advection': AdvectionEquation(advection_speed=float(config.get('advection_speed', 1.0))),
    }
    equation = equation_systems[config['equation']]

    # Set up grid
    grid = gen_grid(config['xmin'], config['xmax'], config['nx'], config['stretch_factor'], config['dimension'])
    if isinstance(grid, tuple):
        grid = np.array(grid[0]).flatten()
    
    n_vars = len(equation.get_variable_names())
    W = np.zeros((n_vars, config['nx']), dtype=float)

    # Set initial conditions
    if config['equation'] == 'euler':
        split_idx = int(config['nx'] * config['initial_conditions']['euler']['split'])
        W[:, :split_idx] = np.array(config['initial_conditions']['euler']['left'], dtype=float)[:, np.newaxis]
        W[:, split_idx:] = np.array(config['initial_conditions']['euler']['right'], dtype=float)[:, np.newaxis]
    elif config['equation'] == 'isentropic':
        split_idx = int(config['nx'] * config['initial_conditions']['isentropic']['split'])
        W[:, :split_idx] = np.array(config['initial_conditions']['isentropic']['left'], dtype=float)[:, np.newaxis]
        W[:, split_idx:] = np.array(config['initial_conditions']['isentropic']['right'], dtype=float)[:, np.newaxis]
    elif config['equation'] == 'shallow_water':
        split_idx = int(config['nx'] * config['initial_conditions']['shallow_water']['split'])
        W[:, :split_idx] = np.array(config['initial_conditions']['shallow_water']['left'], dtype=float)[:, np.newaxis]
        W[:, split_idx:] = np.array(config['initial_conditions']['shallow_water']['right'], dtype=float)[:, np.newaxis]
    elif config['equation'] == 'advection':
        split_idx = int(config['nx'] * config['initial_conditions']['advection']['split'])
        W[:, :split_idx] = np.array(config['initial_conditions']['advection']['left'], dtype=float)[:, np.newaxis]
        W[:, split_idx:] = np.array(config['initial_conditions']['advection']['right'], dtype=float)[:, np.newaxis]

    # Set up boundary conditions (use new argument names)
    bc = BoundaryCondition(
        equation_system=equation,
        bc_kind=config['bc_type'],
        grid=grid,
        left_boundary_state=config['left_values'],
        right_boundary_state=config['right_values']
    )

    # Initialize solver
    solver = Solver(
        equation_system=equation,
        boundary_condition=bc,
        grid=grid,
        cfl=config['cfl'],
        flux=config['flux'],
        reconstruction=config['reconstruction'],
        reconstruct_in_primitive=(config['reconstruction_vars'] == 'primitive'),
        limiter=config['limiter'],
        max_iterations=config['max_iterations'],
        convergence_tol=config['convergence_tolerance'],
        output_filename=config['output_filename']
    )

    # Convert to conservative variables
    U0 = equation.to_conservative_batch(W)

    # Solve and save
    U_history, final_t = solver.solve(U0, config['T'], n_ghost=1)
    print(f"Final simulation time: {final_t:.4f}")
    solver.plot_solution(U_history, final_t, 'u' if config['equation'] == 'advection' else ('height' if config['equation'] == 'shallow_water' else 'density'))

if __name__ == '__main__':
    main()
