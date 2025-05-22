import unittest
import numpy as np
from src.isentropic_gas import IsentropicGasSystem
from src.reconstructions import Reconstruction
from src.boundary_conditions import BoundaryCondition

class TestReconstructions(unittest.TestCase):
    """Unit tests for Reconstruction class methods."""

    def setUp(self):
        """Set up test parameters."""
        self.nx = 12
        self.dx = 1.0 / (self.nx - 1)
        self.isentropic = IsentropicGasSystem(gamma=1.4, k=1.0)
        self.n_ghost = 2
        W = np.zeros((2, self.nx + 2 * self.n_ghost))
        W[0, :] = np.linspace(1.0, 2.0, self.nx + 2 * self.n_ghost)
        W[1, :] = 0.1  # Non-zero velocity for reflective BC test
        self.U = np.array([self.isentropic.to_conservative(W[:, i]) for i in range(self.nx + 2 * self.n_ghost)]).T

    def _to_primitive_array(self, U: np.ndarray) -> np.ndarray:
        """Helper to convert conservative variables to primitive."""
        n_cells = U.shape[1]
        return np.array([self.isentropic.to_primitive(U[:, i]) for i in range(n_cells)]).T

    def test_piecewise_constant(self):
        """Test piecewise constant reconstruction."""
        recon = Reconstruction(self.isentropic)
        bc = BoundaryCondition(self.isentropic, 'periodic')
        U_bc = bc.apply_bcs(self.U, self.n_ghost)
        UL, UR = recon.piecewise_constant(U_bc, self.dx, self.n_ghost)
        for i in range(self.nx - 1):
            np.testing.assert_array_almost_equal(UL[:, i], U_bc[:, i + self.n_ghost], decimal=5)
            np.testing.assert_array_almost_equal(UR[:, i], U_bc[:, i + self.n_ghost + 1], decimal=5)

    def test_muscl_dirichlet(self):
        """Test MUSCL with Dirichlet BCs."""
        bc = BoundaryCondition(self.isentropic, 'dirichlet', left_values=np.array([1.0, 0.0]), right_values=np.array([2.0, 0.0]))
        U_bc = bc.apply_bcs(self.U, self.n_ghost)
        recon = Reconstruction(self.isentropic, limiter='minmod')
        UL, UR = recon.muscl(U_bc, self.dx, self.n_ghost)
        W = self._to_primitive_array(U_bc)
        idx = self.n_ghost
        # Left boundary (density)
        sigma_L = recon.limiter.limit(
            (W[0, idx] - W[0, idx - 1]) / self.dx,
            (W[0, idx + 1] - W[0, idx]) / self.dx
        )
        W_L = W[0, idx] + 0.5 * self.dx * sigma_L
        expected_UL = self.isentropic.to_conservative(np.array([W_L, W[1, idx]]))[0]
        self.assertAlmostEqual(UL[0, 0], expected_UL, places=5)
        # Left boundary (velocity)
        sigma_L_u = recon.limiter.limit(
            (W[1, idx] - W[1, idx - 1]) / self.dx,
            (W[1, idx + 1] - W[1, idx]) / self.dx
        )
        W_L_u = W[1, idx] + 0.5 * self.dx * sigma_L_u
        expected_UL_u = self.isentropic.to_conservative(np.array([W[0, idx], W_L_u]))[1]
        self.assertAlmostEqual(UL[1, 0], expected_UL_u, places=5)

    def test_muscl_neumann(self):
        """Test MUSCL with Neumann BCs (zero gradient)."""
        bc = BoundaryCondition(self.isentropic, 'neumann', left_values=np.array([0.0, 0.0]), right_values=np.array([0.0, 0.0]), dx=self.dx)
        U_bc = bc.apply_bcs(self.U, self.n_ghost)
        recon = Reconstruction(self.isentropic, limiter='minmod')
        UL, UR = recon.muscl(U_bc, self.dx, self.n_ghost)
        W = self._to_primitive_array(U_bc)
        idx = self.n_ghost
        # Left boundary (density)
        sigma_L = recon.limiter.limit(
            (W[0, idx] - W[0, idx - 1]) / self.dx,
            (W[0, idx + 1] - W[0, idx]) / self.dx
        )
        W_L = W[0, idx] + 0.5 * self.dx * sigma_L
        expected_UL = self.isentropic.to_conservative(np.array([W_L, W[1, idx]]))[0]
        self.assertAlmostEqual(UL[0, 0], expected_UL, places=5)
        # Right boundary (density)
        idx_r = self.nx + self.n_ghost - 1
        sigma_R = recon.limiter.limit(
            (W[0, idx_r + 1] - W[0, idx_r]) / self.dx,
            (W[0, idx_r + 2] - W[0, idx_r + 1]) / self.dx
        )
        W_R = W[0, idx_r + 1] - 0.5 * self.dx * sigma_R
        expected_UR = self.isentropic.to_conservative(np.array([W_R, W[1, idx_r + 1]]))[0]
        self.assertAlmostEqual(UR[0, -1], expected_UR, places=5)

    def test_muscl_reflective(self):
        """Test MUSCL with reflective BCs."""
        bc = BoundaryCondition(self.isentropic, 'reflective')
        U_bc = bc.apply_bcs(self.U, self.n_ghost)
        recon = Reconstruction(self.isentropic, limiter='minmod')
        UL, UR = recon.muscl(U_bc, self.dx, self.n_ghost)
        W = self._to_primitive_array(U_bc)
        idx = self.n_ghost
        # Verify ghost cell BCs
        self.assertAlmostEqual(W[1, n_ghost - 1], -W[1, n_ghost], places=5)
        self.assertAlmostEqual(W[0, n_ghost - 1], W[0, n_ghost], places=5)
        # Left boundary (density)
        sigma_L = recon.limiter.limit(
            (W[0, idx] - W[0, idx - 1]) / self.dx,
            (W[0, idx + 1] - W[0, idx]) / self.dx
        )
        W_L = W[0, idx] + 0.5 * self.dx * sigma_L
        expected_UL = self.isentropic.to_conservative(np.array([W_L, W[1, idx]]))[0]
        self.assertAlmostEqual(UL[0, 0], expected_UL, places=5)
        # Left boundary (velocity)
        sigma_L_u = recon.limiter.limit(
            (W[1, idx] - W[1, idx - 1]) / self.dx,
            (W[1, idx + 1] - W[1, idx]) / self.dx
        )
        W_L_u = W[1, idx] + 0.5 * self.dx * sigma_L_u
        expected_UL_u = self.isentropic.to_conservative(np.array([W[0, idx], W_L_u]))[1]
        self.assertAlmostEqual(UL[1, 0], expected_UL_u, places=5)

    def test_ppm_simple(self):
        """Test simple PPM reconstruction."""
        bc = BoundaryCondition(self.isentropic, 'periodic')
        U_bc = bc.apply_bcs(self.U, self.n_ghost)
        recon = Reconstruction(self.isentropic)
        UL, UR = recon.ppm(U_bc, self.dx, self.n_ghost, full_stencil=False)
        W = self._to_primitive_array(U_bc)
        idx = self.n_ghost
        delta_W = 0.5 * (W[0, idx + 1] - W[0, idx - 1])
        W_R = W[0, idx] + delta_W
        expected_UL = self.isentropic.to_conservative(np.array([W_R, W[1, idx]]))[0]
        self.assertAlmostEqual(UL[0, 0], expected_UL, places=5)

    def test_ppm_full(self):
        """Test full PPM reconstruction."""
        bc = BoundaryCondition(self.isentropic, 'periodic')
        U_bc = bc.apply_bcs(self.U, self.n_ghost)
        recon = Reconstruction(self.isentropic)
        UL, UR = recon.ppm(U_bc, self.dx, self.n_ghost, full_stencil=True)
        W = self._to_primitive_array(U_bc)
        idx = self.n_ghost + 2  # Interior cell
        W_R = (7/12) * (W[0, idx] + W[0, idx + 1]) - (1/12) * (W[0, idx - 1] + W[0, idx + 2])
        expected_UL = self.isentropic.to_conservative(np.array([W_R, W[1, idx]]))[0]
        self.assertAlmostEqual(UL[0, 2], expected_UL, places=5)

    def test_weno5(self):
        """Test WENO5 reconstruction."""
        bc = BoundaryCondition(self.isentropic, 'periodic')
        U_bc = bc.apply_bcs(self.U, self.n_ghost)
        recon = Reconstruction(self.isentropic)
        UL, UR = recon.weno5(U_bc, self.dx, self.n_ghost)
        W = self._to_primitive_array(U_bc)
        idx = self.n_ghost
        v = W[0, idx - 2:idx + 3]
        beta0 = 13.0 / 12.0 * (v[0] - 2 * v[1] + v[2])**2 + 0.25 * (v[0] - 4 * v[1] + 3 * v[2])**2
        beta1 = 13.0 / 12.0 * (v[1] - 2 * v[2] + v[3])**2 + 0.25 * (v[1] - v[3])**2
        beta2 = 13.0 / 12.0 * (v[2] - 2 * v[3] + v[4])**2 + 0.25 * (3 * v[2] - 4 * v[3] + v[4])**2
        epsilon = 1e-6
        d0, d1, d2 = 0.1, 0.6, 0.3
        alpha0 = d0 / (beta0 + epsilon)**2
        alpha1 = d1 / (beta1 + epsilon)**2
        alpha2 = d2 / (beta2 + epsilon)**2
        sum_alpha = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / sum_alpha
        omega1 = alpha1 / sum_alpha
        omega2 = alpha2 / sum_alpha
        p0 = (2 * v[0] - 7 * v[1] + 11 * v[2]) / 6.0
        p1 = (-v[1] + 5 * v[2] + 2 * v[3]) / 6.0
        p2 = (2 * v[2] + 5 * v[3] - v[4]) / 6.0
        W_L = omega0 * p0 + omega1 * p1 + omega2 * p2
        expected_UL = self.isentropic.to_conservative(np.array([W_L, W[1, idx]]))[0]
        self.assertAlmostEqual(UL[0, 0], expected_UL, places=5)

    def test_muscl_n_ghost_1(self):
        """Test MUSCL with n_ghost=1."""
        n_ghost = 1
        W = np.zeros((2, self.nx + 2 * n_ghost))
        W[0, :] = np.linspace(1.0, 2.0, self.nx + 2 * n_ghost)
        W[1, :] = 0.1
        U = np.array([self.isentropic.to_conservative(W[:, i]) for i in range(self.nx + 2 * n_ghost)]).T
        bc = BoundaryCondition(self.isentropic, 'reflective')
        U_bc = bc.apply_bcs(U, n_ghost)
        recon = Reconstruction(self.isentropic, limiter='minmod')
        UL, UR = recon.muscl(U_bc, self.dx, n_ghost)
        W = self._to_primitive_array(U_bc)
        idx = n_ghost
        sigma_L = recon.limiter.limit(
            (W[0, idx] - W[0, idx - 1]) / self.dx,
            (W[0, idx + 1] - W[0, idx]) / self.dx
        )
        W_L = W[0, idx] + 0.5 * self.dx * sigma_L
        expected_UL = self.isentropic.to_conservative(np.array([W_L, W[1, idx]]))[0]
        self.assertAlmostEqual(UL[0, 0], expected_UL, places=5)

    def test_all_methods_reflective(self):
        """Test all reconstruction methods with reflective BCs."""
        bc = BoundaryCondition(self.isentropic, 'reflective')
        U_bc = bc.apply_bcs(self.U, self.n_ghost)
        W = self._to_primitive_array(U_bc)
        recon = Reconstruction(self.isentropic, limiter='minmod')

        # Piecewise Constant
        UL, UR = recon.piecewise_constant(U_bc, self.dx, self.n_ghost)
        self.assertAlmostEqual(UL[1, 0], U_bc[1, self.n_ghost], places=5)  # Velocity at left boundary
        self.assertAlmostEqual(UR[1, -1], U_bc[1, self.nx + self.n_ghost], places=5)  # Right boundary

        # MUSCL
        UL, UR = recon.muscl(U_bc, self.dx, self.n_ghost)
        idx = self.n_ghost
        sigma_L_u = recon.limiter.limit(
            (W[1, idx] - W[1, idx - 1]) / self.dx,
            (W[1, idx + 1] - W[1, idx]) / self.dx
        )
        W_L_u = W[1, idx] + 0.5 * self.dx * sigma_L_u
        expected_UL_u = self.isentropic.to_conservative(np.array([W[0, idx], W_L_u]))[1]
        self.assertAlmostEqual(UL[1, 0], expected_UL_u, places=5)

        # PPM Simple
        UL, UR = recon.ppm(U_bc, self.dx, self.n_ghost, full_stencil=False)
        delta_W = 0.5 * (W[1, idx + 1] - W[1, idx - 1])
        W_R = W[1, idx] + delta_W
        expected_UL = self.isentropic.to_conservative(np.array([W[0, idx], W_R]))[1]
        self.assertAlmostEqual(UL[1, 0], expected_UL, places=5)

        # PPM Full
        UL, UR = recon.ppm(U_bc, self.dx, self.n_ghost, full_stencil=True)
        idx = self.n_ghost + 2  # Interior cell
        W_R = (7/12) * (W[1, idx] + W[1, idx + 1]) - (1/12) * (W[1, idx - 1] + W[1, idx + 2])
        expected_UL = self.isentropic.to_conservative(np.array([W[0, idx], W_R]))[1]
        self.assertAlmostEqual(UL[1, 2], expected_UL, places=5)

        # WENO5
        UL, UR = recon.weno5(U_bc, self.dx, self.n_ghost)
        v = W[1, idx - 2:idx + 3]
        beta0 = 13.0 / 12.0 * (v[0] - 2 * v[1] + v[2])**2 + 0.25 * (v[0] - 4 * v[1] + 3 * v[2])**2
        beta1 = 13.0 / 12.0 * (v[1] - 2 * v[2] + v[3])**2 + 0.25 * (v[1] - v[3])**2
        beta2 = 13.0 / 12.0 * (v[2] - 2 * v[3] + v[4])**2 + 0.25 * (3 * v[2] - 4 * v[3] + v[4])**2
        epsilon = 1e-6
        d0, d1, d2 = 0.1, 0.6, 0.3
        alpha0 = d0 / (beta0 + epsilon)**2
        alpha1 = d1 / (beta1 + epsilon)**2
        alpha2 = d2 / (beta2 + epsilon)**2
        sum_alpha = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / sum_alpha
        omega1 = alpha1 / sum_alpha
        omega2 = alpha2 / sum_alpha
        p0 = (2 * v[0] - 7 * v[1] + 11 * v[2]) / 6.0
        p1 = (-v[1] + 5 * v[2] + 2 * v[3]) / 6.0
        p2 = (2 * v[2] + 5 * v[3] - v[4]) / 6.0
        W_L = omega0 * p0 + omega1 * p1 + omega2 * p2
        expected_UL = self.isentropic.to_conservative(np.array([W[0, idx], W_L]))[1]
        self.assertAlmostEqual(UL[1, 0], expected_UL, places=5)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        recon = Reconstruction(self.isentropic)
        bc = BoundaryCondition(self.isentropic, 'periodic')
        U_wrong_shape = np.zeros((3, self.nx + 2 * self.n_ghost))  # Wrong n_vars
        with self.assertRaises(ValueError):
            recon.piecewise_constant(U_wrong_shape, self.dx, self.n_ghost)
        U_too_small = np.zeros((2, self.n_ghost))  # Too few cells
        with self.assertRaises(ValueError):
            recon.piecewise_constant(U_too_small, self.dx, self.n_ghost)
        with self.assertRaises(ValueError):
            recon.weno5(self.U, self.dx, n_ghost=1)  # WENO5 needs n_ghost >= 2
        bc_wrong = BoundaryCondition(self.isentropic, 'dirichlet', left_values=np.array([1.0]))  # Wrong shape
        with self.assertRaises(ValueError):
            bc_wrong.apply_bcs(self.U, self.n_ghost)

if __name__ == '__main__':
    unittest.main()