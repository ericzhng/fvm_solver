import argparse
import numpy as np
from src.isentropic_gas import IsentropicGasSystem
from src.shallow_water import ShallowWaterSystem
from src.euler import EulerEquationSystem
from src.solver import Solver


def parse_args():
    """Parse command-line arguments for the solver.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='1D Hyperbolic Conservation Law Solver')
    parser.add_argument('--equation', type=str, default='isentropic',
                        choices=['isentropic', 'shallow_water', 'euler'],
                        help='Equation system to solve')
    parser.add_argument('--nx', type=int, default=100, help='Number of grid points')
    parser.add_argument('--T', type=float, default=0.2, help='Final simulation time')
    parser.add_argument('--cfl', type=float, default=0.5, help='CFL number')
    parser.add_argument('--flux', type=str, default='HLLC',
                        choices=['lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe'],
                        help='Numerical flux method')
    parser.add_argument('--reconstruction', type=str, default='piecewise_constant',
                        choices=['piecewise_constant', 'muscl', 'ppm', 'weno5'],
                        help='Reconstruction method')
    parser.add_argument('--bc_type', type=str, default='transmissive',
                        choices=['periodic', 'reflective', 'transmissive'],
                        help='Boundary condition type')
    parser.add_argument('--output', type=str, default='solution.png', help='Output plot filename')
    return parser.parse_args()


def main():
    """Run the solver with specified parameters and plot results."""
    args = parse_args()

    # Select equation system
    equation_systems = {
        'isentropic': IsentropicGasSystem(gamma=1.4, k=1.0),
        'shallow_water': ShallowWaterSystem(g=9.81),
        'euler': EulerEquationSystem(gamma=1.4)
    }
    equation = equation_systems[args.equation]

    # Initialize solver
    solver = Solver(
        equation_system=equation,
        flux=args.flux,
        reconstruction=args.reconstruction,
        cfl=args.cfl,
        bc_type=args.bc_type
    )

    # Set up grid
    x = np.linspace(0, 1, args.nx)
    n_vars = len(equation.get_variable_names())
    W = np.zeros((n_vars, args.nx))

    # Set initial conditions based on equation system
    if args.equation == 'isentropic':
        W[0, :args.nx//2] = 2.0  # Higher density left
        W[0, args.nx//2:] = 1.0  # Lower density right
        W[1, :] = 0.0  # Zero velocity
    elif args.equation == 'shallow_water':
        W[0, :] = 1.0  # Constant height
        W[1, :args.nx//2] = 0.5  # Positive velocity left
        W[1, args.nx//2:] = -0.5  # Negative velocity right
    else:  # euler
        W[0, :] = 1.0  # Baseline density
        W[1, :] = 0.0  # Zero velocity
        W[2, :] = 1.0  # Constant pressure
        W[0, :args.nx//2] = 2.0  # Higher density left

    # Convert to conservative variables
    U0 = np.array([equation.to_conservative(W[:, i]) for i in range(args.nx)]).T

    # Solve and plot
    U_history, final_t = solver.solve(U0, x, args.T, n_ghost=2)
    solver.plot_solution(U_history, x, args.T, filename=args.output)


if __name__ == '__main__':
    main()