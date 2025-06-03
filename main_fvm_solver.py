import argparse
import numpy as np

from src.utils import parse_xml_config, gen_grid  # Make sure this import path matches your project structure

from src.boundary import BoundaryCondition
from src.equation.isentropic_gas_equation import IsentropicGas
from src.equation.shallow_water_equation import ShallowWater
from src.equation.euler_equation import EulerEquation
from src.solver import Solver

def main():
    """
    Run the finite volume solver with XML configuration and save results.

    Reads configuration, sets up the equation system, grid, initial and boundary conditions,
    runs the solver, and plots the final solution snapshot.
    """
    parser = argparse.ArgumentParser(description='Finite Volume Riemann Solver for 1D/2D/3D Hyperbolic Conservation Laws')
    parser.add_argument('--config', type=str, default='input_config.xml', help='Path to XML configuration file')
    args = parser.parse_args()

    config = parse_xml_config(args.config)

    # Select equation system
    equation_systems = {
        'euler': EulerEquation(gamma=config['gamma']),
        'isentropic': IsentropicGas(gamma=config['gamma'], k=1.0),
        'shallow_water': ShallowWater(g=9.81),
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
    U_history, final_t = solver.solve(U0, config['T'], n_ghost=2)
    print(f"Final simulation time: {final_t:.4f}")
    solver.plot_solution(U_history, final_t, 'density')

if __name__ == '__main__':
    main()
