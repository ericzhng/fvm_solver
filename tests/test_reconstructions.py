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

if __name__ == '__main__':
    unittest.main()
    