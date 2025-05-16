import unittest
import numpy as np
from src.isentropic_gas import IsentropicGasSystem
from src.shallow_water import ShallowWaterSystem
from src.euler import EulerEquationSystem
from src.solver import Solver


class TestSolver(unittest.TestCase):
    """Unit tests for Solver class methods."""

    def setUp(self):
        """Set up test parameters for solver tests."""
        self.nx = 10  # Number of grid cells
        self.x = np.linspace(0, 1, self.nx)  # Spatial grid
        self.dx = self.x[1] - self.x[0]  # Grid spacing
        self.n_ghost = 2  # Number of ghost cells per side
        self.isentropic = IsentropicGasSystem(gamma=1.4, k=1.0)
        self.shallow_water = ShallowWaterSystem(g=9.81)
        self.euler = EulerEquationSystem(gamma=1.4)
        self.solver = Solver(
            equation_system=self.isentropic,
            flux='Roe',
            reconstruction='piecewise_constant',
            cfl=0.5,
            bc_type='periodic'
        )
        # Initialize primitive variables: constant density, zero velocity
        W = np.zeros((2, self.nx))
        W[0, :] = 1.0  # density
        W[1, :] = 0.0  # velocity
        self.U0 = np.array([self.isentropic.to_conservative(W[:, i]) for i in range(self.nx)]).T

    def test_compute_dt(self):
        """Test time step computation based on CFL condition."""
        U = self.U0.copy()
        dt = self.solver.compute_dt(U, self.dx)
        W = np.array([self.isentropic.to_primitive(U[:, i]) for i in range(self.nx)]).T
        max_speed = np.max(np.abs(W[1]) + self.isentropic.sound_speed(W))
        expected_dt = 0.5 * self.dx / max_speed
        self.assertAlmostEqual(dt, expected_dt, places=5)

    def test_periodic_bc(self):
        """Test periodic boundary condition application."""
        U = self.U0.copy()
        U_ext = self.solver.apply_periodic_bc(U, self.n_ghost)
        # Check ghost cells copy from opposite ends
        np.testing.assert_array_almost_equal(U_ext[:, :self.n_ghost], U[:, -self.n_ghost:], decimal=5)
        np.testing.assert_array_almost_equal(U_ext[:, -self.n_ghost:], U[:, :self.n_ghost], decimal=5)
        np.testing.assert_array_almost_equal(U_ext[:, self.n_ghost:-self.n_ghost], U, decimal=5)

    def test_reflective_bc(self):
        """Test reflective boundary condition application."""
        solver = Solver(
            equation_system=self.isentropic,
            flux='Roe',
            reconstruction='piecewise_constant',
            cfl=0.5,
            bc_type='reflective'
        )
        U = self.U0.copy()
        U[1, :] = 1.0  # Non-zero velocity
        U_ext = solver.apply_reflective_bc(U, self.n_ghost)
        for i in range(self.n_ghost):
            # Left boundary: mirror with negated velocity
            expected = U[:, self.n_ghost - 1 - i].copy()
            expected[1] = -expected[1]
            np.testing.assert_array_almost_equal(U_ext[:, i], expected, decimal=5)
            # Right boundary: mirror with negated velocity
            expected = U[:, -i - 1].copy()
            expected[1] = -expected[1]
            np.testing.assert_array_almost_equal(U_ext[:, -self.n_ghost + i], expected, decimal=5)

    def test_transmissive_bc(self):
        """Test transmissive boundary condition application."""
        solver = Solver(
            equation_system=self.isentropic,
            flux='Roe',
            reconstruction='piecewise_constant',
            cfl=0.5,
            bc_type='transmissive'
        )
        U = self.U0.copy()
        U_ext = solver.apply_transmissive_bc(U, self.n_ghost)
        for i in range(self.n_ghost):
            # Left ghost cells copy left boundary
            np.testing.assert_array_almost_equal(U_ext[:, i], U[:, 0], decimal=5)
            # Right ghost cells copy right boundary
            np.testing.assert_array_almost_equal(U_ext[:, -self.n_ghost + i], U[:, -1], decimal=5)

    def test_solver_isentropic(self):
        """Test solver with IsentropicGasSystem on a Riemann problem."""
        solver = Solver(
            equation_system=self.isentropic,
            flux='Roe',
            reconstruction='piecewise_constant',
            cfl=0.5,
            bc_type='transmissive'
        )
        # Riemann problem: density jump
        W = np.zeros((2, self.nx))
        W[0, :self.nx//2] = 2.0  # Higher density left
        W[0, self.nx//2:] = 1.0  # Lower density right
        W[1, :] = 0.0  # Zero velocity
        U0 = np.array([self.isentropic.to_conservative(W[:, i]) for i in range(self.nx)]).T
        U_history, final_t = solver.solve(U0, self.x, T=0.1, n_ghost=self.n_ghost)
        W_final = np.array([self.isentropic.to_primitive(U_history[-1][:, i]) for i in range(self.nx)]).T
        # Check physical consistency
        self.assertTrue(np.all(W_final[0, :] > 0))  # Positive density
        # Check boundary consistency for transmissive BC
        self.assertTrue(np.allclose(W_final[0, 0], W_final[0, 1], atol=1e-5))
        self.assertTrue(np.allclose(W_final[0, -1], W_final[0, -2], atol=1e-5))

    def test_solver_shallow_water(self):
        """Test solver with ShallowWaterSystem on a dam-break problem."""
        solver = Solver(
            equation_system=self.shallow_water,
            flux='HLLC',
            reconstruction='piecewise_constant',
            cfl=0.5,
            bc_type='reflective'
        )
        # Dam-break: velocity discontinuity
        W = np.zeros((2, self.nx))
        W[0, :] = 1.0  # Constant height
        W[1, :self.nx//2] = 0.5  # Positive velocity left
        W[1, self.nx//2:] = -0.5  # Negative velocity right
        U0 = np.array([self.shallow_water.to_conservative(W[:, i]) for i in range(self.nx)]).T
        U_history, final_t = solver.solve(U0, self.x, T=0.1, n_ghost=self.n_ghost)
        W_final = np.array([self.shallow_water.to_primitive(U_history[-1][:, i]) for i in range(self.nx)]).T
        # Check physical consistency
        self.assertTrue(np.all(W_final[0, :] > 0))  # Positive height
        # Check reflective BC: zero velocity at boundaries
        self.assertAlmostEqual(W_final[1, 0], 0.0, places=5)
        self.assertAlmostEqual(W_final[1, -1], 0.0, places=5)

    def test_solver_euler(self):
        """Test solver with EulerEquationSystem on a shock tube problem."""
        solver = Solver(
            equation_system=self.euler,
            flux='Roe',
            reconstruction='piecewise_constant',
            cfl=0.5,
            bc_type='periodic'
        )
        # Shock tube: density jump
        W = np.zeros((3, self.nx))
        W[0, :] = 1.0  # Baseline density
        W[1, :] = 0.0  # Zero velocity
        W[2, :] = 1.0  # Constant pressure
        W[0, :self.nx//2] = 2.0  # Higher density left
        U0 = np.array([self.euler.to_conservative(W[:, i]) for i in range(self.nx)]).T
        U_history, final_t = solver.solve(U0, self.x, T=0.1, n_ghost=self.n_ghost)
        W_final = np.array([self.euler.to_primitive(U_history[-1][:, i]) for i in range(self.nx)]).T
        # Check physical consistency
        self.assertTrue(np.all(W_final[0, :] > 0))  # Positive density
        self.assertTrue(np.all(W_final[2, :] > 0))  # Positive pressure
        # Check periodic BC: states match at boundaries
        self.assertTrue(np.allclose(W_final[:, 0], W_final[:, -1], atol=1e-5))


if __name__ == '__main__':
    unittest.main()