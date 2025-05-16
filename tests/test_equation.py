import unittest
import numpy as np
from src.isentropic_gas import IsentropicGasSystem
from src.shallow_water import ShallowWaterSystem
from src.euler import EulerEquationSystem


class TestEquationSystems(unittest.TestCase):
    """Unit tests for EquationSystem and its subclasses."""

    def setUp(self):
        """Set up equation systems for testing."""
        self.isentropic = IsentropicGasSystem(gamma=1.4, k=1.0)
        self.shallow_water = ShallowWaterSystem(g=9.81)
        self.euler = EulerEquationSystem(gamma=1.4)

    def test_isentropic_conversions(self):
        """Test conversions between primitive and conservative variables for IsentropicGasSystem."""
        W = np.array([1.0, 0.5])  # [density, velocity]
        U = self.isentropic.to_conservative(W)
        np.testing.assert_array_almost_equal(U, [1.0, 0.5], decimal=5)
        W_back = self.isentropic.to_primitive(U)
        np.testing.assert_array_almost_equal(W_back, W, decimal=5)

    def test_isentropic_flux(self):
        """Test flux computation for IsentropicGasSystem."""
        W = np.array([1.0, 0.5])  # [density, velocity]
        U = self.isentropic.to_conservative(W)
        F = self.isentropic.compute_flux(U, W)
        p = self.isentropic.k * W[0]**self.isentropic.gamma  # Pressure
        expected = np.array([W[0] * W[1], W[0] * W[1]**2 + p])
        np.testing.assert_array_almost_equal(F, expected, decimal=5)

    def test_isentropic_sound_speed(self):
        """Test sound speed computation for IsentropicGasSystem."""
        W = np.array([1.0, 0.0])  # [density, velocity]
        c = self.isentropic.sound_speed(W)
        expected = np.sqrt(self.isentropic.gamma * self.isentropic.k * W[0]**(self.isentropic.gamma - 1))
        self.assertAlmostEqual(c, expected, places=5)

    def test_shallow_water_conversions(self):
        """Test conversions between primitive and conservative variables for ShallowWaterSystem."""
        W = np.array([1.0, 0.5])  # [height, velocity]
        U = self.shallow_water.to_conservative(W)
        np.testing.assert_array_almost_equal(U, [1.0, 0.5], decimal=5)
        W_back = self.shallow_water.to_primitive(U)
        np.testing.assert_array_almost_equal(W_back, W, decimal=5)

    def test_euler_conversions(self):
        """Test conversions between primitive and conservative variables for EulerEquationSystem."""
        W = np.array([1.0, 0.5, 1.0])  # [density, velocity, pressure]
        U = self.euler.to_conservative(W)
        E = W[2] / (self.euler.gamma - 1) + 0.5 * W[0] * W[1]**2  # Total energy
        np.testing.assert_array_almost_equal(U, [W[0], W[0] * W[1], E], decimal=5)
        W_back = self.euler.to_primitive(U)
        np.testing.assert_array_almost_equal(W_back, W, decimal=5)


if __name__ == '__main__':
    unittest.main()