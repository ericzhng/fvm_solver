import argparse
import numpy as np
from src.boundary import BoundaryCondition
from src.equation.isentropic_gas_equation import IsentropicGas
from src.equation.shallow_water_equation import ShallowWater
from src.equation.euler_equation import EulerEquation
from src.solver import Solver


def generate_non_uniform_grid(xmin, xmax, nx, stretch_factor=1.0):
    x = np.zeros(nx + 1)
    for i in range(nx + 1):
        x[i] = xmin + (xmax - xmin) * (1 - np.cos(np.pi * i / nx)) / 2 * stretch_factor
    return x

def parse_args():
    """Parse command-line arguments for the solver.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Finite Volumne based Riemann Solver for 1D Hyperbolic Conservation Law')
    parser.add_argument('--equation', type=str, default='euler', choices=['isentropic', 'shallow_water', 'euler'], help='specify the equation system to solve')
    parser.add_argument('--nx', type=int, default=100, help='Number of grid points')
    parser.add_argument('--T', type=float, default=0.1, help='Final simulation time')
    parser.add_argument('--cfl', type=float, default=0.4, help='CFL number')
    parser.add_argument('--flux', type=str, default='roe', choices=['lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe'], help='Numerical flux method')
    parser.add_argument('--reconstruction', type=str, default='muscl', choices=['piecewise_constant', 'muscl', 'ppm', 'weno5'], help='Reconstruction method')
    parser.add_argument('--reconstruction_vars', type=str, default='conservative', choices=['primitive', 'conservative'], help='Variable type for reconstruction')
    parser.add_argument('--limiter', type=str, default='superbee', choices=['minmod', 'superbee', 'vanleer', 'mc', 'koren', 'osher', 'sweby', 'umist', 'none'], help='limiter method')
    return parser.parse_args()

def main():
    """Run the solver with specified parameters and plot results."""
    args = parse_args()

    bc_type = 'neumann'

    # Select equation system
    equation_systems = {
        'euler': EulerEquation(gamma=1.4),
        'isentropic': IsentropicGas(gamma=1.4, k=1.0),
        'shallow_water': ShallowWater(g=9.81),
    }
    equation = equation_systems[args.equation]

    # Set up grid
    x = generate_non_uniform_grid(0, 1, args.nx, stretch_factor=1.0)
    
    n_vars = len(equation.get_variable_names())
    W = np.zeros((n_vars, args.nx))

    # Set up boundary conditions
    if bc_type == 'neumann':
        left_values = np.zeros(equation.n_vars)
        right_values = np.zeros(equation.n_vars)
    elif bc_type == 'dirichlet':
        left_values = np.array([2.0, 0.0])  # Example Dirichlet values
        right_values = np.array([1.0, 0.0])
    else:
        left_values = np.zeros(equation.n_vars)
        right_values = np.zeros(equation.n_vars)
    
    bc2apply = BoundaryCondition(equation, bc_type.lower(), x, left_values=left_values, right_values=right_values)

    if args.reconstruction_vars == 'primitive':
        use_primitive = True
    else:
        use_primitive = False

    # Initialize solver
    solver = Solver(
        equation_system=equation,
        boundary_condition=bc2apply,
        cfl=args.cfl,
        flux=args.flux,
        reconstruction=args.reconstruction,
        use_primitive_reconstruction=use_primitive,
        limiter='minmod' if args.reconstruction == 'muscl' else 'none'
    )

    totalT = args.T

    # Set initial conditions based on equation system
    if args.equation == 'isentropic':
        W[0, :args.nx//2] = 2.0  # Higher density left
        W[0, args.nx//2:] = 1.0  # Lower density right
        W[1, :] = 0.0  # Zero velocity

    elif args.equation == 'shallow_water':
        W[0, :] = 1.0  # Constant height
        W[1, :args.nx//2] = 0.5  # Positive velocity left
        W[1, args.nx//2:] = -0.5  # Negative velocity right

    elif args.equation == 'euler':  # euler
        W[0, :] = 0.125  # Baseline density
        W[1, :] = 0.0  # Zero velocity
        W[2, :] = 0.1  # Constant pressure
        W[0, :args.nx//2] = 1.0  # Higher density left
        W[2, :args.nx//2] = 1.0  # Higher pressure left

    # Convert to conservative variables
    U0 = np.array([equation.to_conservative(W[:, i]) for i in range(args.nx)]).T

    # Solve and plot
    U_history, final_t = solver.solve(U0, x, totalT, n_ghost=2)
    print(f"Final simulation time reached: {final_t:.4f}")
    solver.plot_solution(U_history, x, totalT, 'height' if args.equation == 'shallow_water' else 'density')

if __name__ == '__main__':
    main()
