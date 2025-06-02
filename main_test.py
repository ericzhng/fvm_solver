import argparse
import numpy as np
import xml.etree.ElementTree as ET
from typing import Union
from src.boundary import BoundaryCondition
from src.equation.isentropic_gas_equation import IsentropicGas
from src.equation.shallow_water_equation import ShallowWater
from src.equation.euler_equation import EulerEquation
from src.solver import Solver

def generate_non_uniform_grid(xmin, xmax, nx, stretch_factor=1.0, dim=1) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
    """Generate a non-uniform grid for 1D, 2D, or 3D domains.

    Args:
        xmin (float): Minimum coordinate.
        xmax (float): Maximum coordinate.
        nx (int): Number of grid points.
        stretch_factor (float): Grid stretching factor.
        dim (int): Dimension of the grid (1, 2, or 3).

    Returns:
        np.ndarray: Grid coordinates.
    """
    x = np.zeros(nx + 1)
    for i in range(nx + 1):
        x[i] = xmin + (xmax - xmin) * (1 - np.cos(np.pi * i / nx)) / 2 * stretch_factor
    if dim == 1:
        return x
    elif dim == 2:
        return np.meshgrid(x, x)
    elif dim == 3:
        return np.meshgrid(x, x, x)
    else:
        raise ValueError("Dimension must be 1, 2, or 3")

def get_text(element, default=None, required=False, cast=None):
    if element is not None and element.text is not None:
        try:
            return cast(element.text) if cast else element.text
        except Exception:
            if required:
                raise ValueError(f"Invalid value for element {element.tag}")
            return default
    if required:
        raise ValueError(f"Missing required element or text for {element.tag if element is not None else 'unknown'}")
    return default

def parse_xml_config(filename):
    """Parse simulation configuration from XML file.

    Args:
        filename (str): Path to XML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    config = {}

    config['dimension'] = get_text(root.find('geometry/dimension'), default=1, required=True, cast=int)
    config['xmin'] = get_text(root.find('geometry/domain/xmin'), default=0.0, required=True, cast=float)
    config['xmax'] = get_text(root.find('geometry/domain/xmax'), default=1.0, required=True, cast=float)
    config['nx'] = get_text(root.find('mesh/nx'), default=100, required=True, cast=int)
    config['stretch_factor'] = get_text(root.find('mesh/stretch_factor'), default=1.0, cast=float)
    config['equation'] = get_text(root.find('equation/type'), default='euler')
    config['gamma'] = get_text(root.find('equation/gamma'), default=1.4, cast=float) if config['equation'] == 'euler' else 1.4
    config['bc_type'] = get_text(root.find('boundary_conditions/type'), default='dirichlet')

    left_values_elem = root.find('boundary_conditions/left_values')
    config['left_values'] = np.array([float(v.text) for v in left_values_elem if v.text is not None] if left_values_elem is not None else [])

    right_values_elem = root.find('boundary_conditions/right_values')
    config['right_values'] = np.array([float(v.text) for v in right_values_elem if v.text is not None] if right_values_elem is not None else [])

    ic_euler_left_elem = root.find('initial_conditions/euler/left')
    ic_euler_right_elem = root.find('initial_conditions/euler/right')
    ic_euler_split_elem = root.find('initial_conditions/euler/split')
    config['initial_conditions'] = {
        'euler': {
            'left': np.array([float(v.text) for v in ic_euler_left_elem if v.text is not None] if ic_euler_left_elem is not None else []),
            'right': np.array([float(v.text) for v in ic_euler_right_elem if v.text is not None] if ic_euler_right_elem is not None else []),
            'split': float(ic_euler_split_elem.text) if ic_euler_split_elem is not None and ic_euler_split_elem.text is not None else 0.5
        }
    }
    config['T'] = get_text(root.find('solver_settings/T'), default=1.0, cast=float)
    config['cfl'] = get_text(root.find('solver_settings/cfl'), default=0.5, cast=float)
    config['flux'] = get_text(root.find('solver_settings/flux'), default='roe')
    config['reconstruction'] = get_text(root.find('solver_settings/reconstruction'), default='linear')
    config['reconstruction_vars'] = get_text(root.find('solver_settings/reconstruction_vars'), default='primitive')
    config['limiter'] = get_text(root.find('solver_settings/limiter'), default='minmod')
    config['max_iterations'] = get_text(root.find('solver_settings/max_iterations'), default=10000, cast=int)
    config['convergence_tolerance'] = get_text(root.find('solver_settings/convergence_tolerance'), default=1e-6, cast=float)
    config['output_format'] = get_text(root.find('output/format'), default='csv')
    config['output_filename'] = get_text(root.find('output/filename'), default='output.csv')

    return config

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
    grid = generate_non_uniform_grid(config['xmin'], config['xmax'], config['nx'], config['stretch_factor'], config['dimension'])
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
