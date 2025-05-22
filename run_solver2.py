import numpy as np
from src.solver import Solver
from src.euler import EulerEquationSystem
from src.isentropic_gas import IsentropicGasSystem
from src.shallow_water import ShallowWaterSystem


def riemann_problem(equation_system, n_cells=100, x_range=(-0.5, 0.5), T=0.2):
    """Set up and solve a Riemann problem for the given equation system."""
    x = np.linspace(x_range[0], x_range[1], n_cells)
    dx = x[1] - x[0]
    U0 = np.zeros((equation_system.n_vars, n_cells))

    # Initialize Riemann problem (left and right states)
    if isinstance(equation_system, EulerEquationSystem):
        WL = np.array([1.0, 0.75, 1.0])  # rho, u, p
        WR = np.array([0.125, 0.0, 0.1])
    elif isinstance(equation_system, IsentropicGasSystem):
        WL = np.array([1.0, 0.75])  # rho, u
        WR = np.array([0.125, 0.0])
    else:  # ShallowWaterSystem
        WL = np.array([1.0, 0.75])  # h, u
        WR = np.array([0.125, 0.0])

    for i in range(n_cells):
        U0[:, i] = equation_system.to_conservative(WL if x[i] < 0 else WR)

    # Set up solver
    solver = Solver(
        equation_system=equation_system,
        flux='hllc',
        reconstruction='weno5',
        cfl=0.5,
        bc_type='dirichlet'
    )

    # Solve
    history, t = solver.solve(U0, x, T, n_ghost=2)
    solver.plot_solution(history, x, t)


if __name__ == "__main__":
    # Test Euler equations
    euler = EulerEquationSystem(gamma=1.4)
    riemann_problem(euler, n_cells=200, T=0.2)

    # Test isentropic gas
    isentropic = IsentropicGasSystem(gamma=1.4, k=1.0)
    riemann_problem(isentropic, n_cells=200, T=0.2)

    # Test shallow water
    shallow = ShallowWaterSystem(g=9.81)
    riemann_problem(shallow, n_cells=200, T=0.2)