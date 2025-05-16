import unittest
import numpy as np
from src.isentropic_gas import IsentropicGasSystem
from src.reconstructions import Reconstruction


class TestReconstructions(unittest.TestCase):
    """Unit tests for Reconstruction class methods."""

    def setUp(self):
        """Set up test parameters for reconstruction tests."""
        self.nx = 10  # Number of grid cells
        self.dx = 1.0 / (self.nx - 1)  # Grid spacing
        self.isentropic = IsentropicGasSystem(gamma=1.4, k=1.0)
        # Initialize primitive variables: constant density, zero velocity
        W = np.zeros((2, self.nx))
        W[0, :] = 1.0  # density
        W[1, :] = 0.0  # velocity
        self.U = np.array([self.isentropic.to_conservative(W[:, i]) for i in range(self.nx)]).T

    def test_piecewise_constant(self):
        """Test piecewise constant reconstruction."""
        recon = Reconstruction(self.isentropic)
        UL, UR = recon.piecewise_constant(self.U, self.dx)
        for i in range(self.nx - 1):
            # Left state should match current cell
            np.testing.assert_array_almost_equal(UL[:, i], self.U[:, i], decimal=5)
            # Right state should match next cell
            np.testing.assert_array_almost_equal(UR[:, i], self.U[:, i + 1], decimal=5)

    def test_muscl(self):
        """Test MUSCL reconstruction with minmod limiter."""
        recon = Reconstruction(self.isentropic, limiter='minmod')
        U = self.U.copy()
        # Linear density profile for testing slopes
        U[0, :] = np.linspace(1.0, 2.0, self.nx)
        UL, UR = recon.muscl(U, self.dx)
        W = np.array([self.isentropic.to_primitive(U[:, i]) for i in range(self.nx)]).T
        for i in range(2, self.nx - 2):
            # Compute limited slope for density
            sigma_L = recon.limiter.limit(
                (W[0, i] - W[0, i - 1]) / self.dx,
                (W[0, i - 1] - W[0, i - 2]) / self.dx
            )
            W_L = W[0, i] - 0.5 * self.dx * sigma_L
            expected_UL = self.isentropic.to_conservative(np.array([W_L, W[1, i]]))[0]
            self.assertAlmostEqual(UL[0, i - 1], expected_UL, places=5)

    def test_ppm(self):
        """Test PPM reconstruction."""
        recon = Reconstruction(self.isentropic)
        U = self.U.copy()
        U[0, :] = np.linspace(1.0, 2.0, self.nx)  # Linear density profile
        UL, UR = recon.ppm(U, self.dx)
        W = np.array([self.isentropic.to_primitive(U[:, i]) for i in range(self.nx)]).T
        for i in range(2, self.nx - 2):
            # Simplified PPM: average of adjacent cells
            W_avg = 0.5 * (W[0, i + 1] + W[0, i])
            expected_UL = self.isentropic.to_conservative(np.array([W_avg, W[1, i]]))[0]
            self.assertAlmostEqual(UL[0, i - 1], expected_UL, places=5)

    def test_weno5(self):
        """Test WENO5 reconstruction."""
        recon = Reconstruction(self.isentropic)
        U = self.U.copy()
        U[0, :] = np.linspace(1.0, 2.0, self.nx)  # Linear density profile
        UL, UR = recon.weno5(U, self.dx)
        W = np.array([self.isentropic.to_primitive(U[:, i]) for i in range(self.nx)]).T
        for i in range(2, self.nx - 2):
            v = W[0, i - 2:i + 3]
            # Smoothness indicators
            beta0 = 13.0 / 12.0 * (v[0] - 2 * v[1] + v[2])**2 + 0.25 * (v[0] - 4 * v[1] + 3 * v[2])**2
            beta1 = 13.0 / 12.0 * (v[1] - 2 * v[2] + v[3])**2 + 0.25 * (v[1] - v[3])**2
            beta2 = 13.0 / 12.0 * (v[2] - 2 * v[3] + v[4])**2 + 0.25 * (3 * v[2] - 4 * v[3] + v[4])**2
            epsilon = max(1e-6, 1e-10 * np.max(np.abs(W)))  # Regularization
            # Nonlinear weights
            d0, d1, d2 = 0.1, 0.6, 0.3  # Ideal weights
            alpha0 = d0 / (beta0 + epsilon)**2
            alpha1 = d1 / (beta1 + epsilon)**2
            alpha2 = d2 / (beta2 + epsilon)**2
            sum_alpha = alpha0 + alpha1 + alpha2
            omega0 = alpha0 / sum_alpha
            omega1 = alpha1 / sum_alpha
            omega2 = alpha2 / sum_alpha
            # Polynomial reconstruction for right state
            p0 = (2 * v[0] - 7 * v[1] + 11 * v[2]) / 6.0
            p1 = (-v[1] + 5 * v[2] + 2 * v[3]) / 6.0
            p2 = (2 * v[2] + 5 * v[3] - v[4]) / 6.0
            W_R = omega0 * p0 + omega1 * p1 + omega2 * p2
            expected_UR = self.isentropic.to_conservative(np.array([W_R, W[1, i]]))[0]
            self.assertAlmostEqual(UR[0, i - 1], expected_UR, places=5)


if __name__ == '__main__':
    unittest.main()