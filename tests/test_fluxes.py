import unittest
import numpy as np
from src.isentropic_gas import IsentropicGasSystem
from src.fluxes import Flux


class TestFluxes(unittest.TestCase):
    """Unit tests for Flux class methods."""

    def setUp(self):
        """Set up test parameters for flux calculations."""
        self.isentropic = IsentropicGasSystem(gamma=1.4, k=1.0)
        self.WL = np.array([1.0, 0.0])  # Left state: [density, velocity]
        self.WR = np.array([0.5, 0.0])  # Right state: [density, velocity]
        self.UL = self.isentropic.to_conservative(self.WL)
        self.UR = self.isentropic.to_conservative(self.WR)

    def test_lax_friedrichs_flux(self):
        """Test Lax-Friedrichs flux computation."""
        flux = Flux(self.isentropic, lambda_max=1.0)
        F = flux.lax_friedrichs(self.UL, self.UR, self.WL, self.WR)
        FL = self.isentropic.compute_flux(self.UL, self.WL)
        FR = self.isentropic.compute_flux(self.UR, self.WR)
        expected = 0.5 * (FL + FR - 1.0 * (self.UR - self.UL))
        np.testing.assert_array_almost_equal(F, expected, decimal=5)

    def test_rusanov_flux(self):
        """Test Rusanov (local Lax-Friedrichs) flux computation."""
        flux = Flux(self.isentropic)
        F = flux.rusanov(self.UL, self.UR, self.WL, self.WR)
        lambda_local = max(
            abs(self.WL[1]) + self.isentropic.sound_speed(self.WL),
            abs(self.WR[1]) + self.isentropic.sound_speed(self.WR)
        )
        FL = self.isentropic.compute_flux(self.UL, self.WL)
        FR = self.isentropic.compute_flux(self.UR, self.WR)
        expected = 0.5 * (FL + FR - lambda_local * (self.UR - self.UL))
        np.testing.assert_array_almost_equal(F, expected, decimal=5)

    def test_force_flux(self):
        """Test FORCE flux computation."""
        flux = Flux(self.isentropic)
        F = flux.force(self.UL, self.UR, self.WL, self.WR)
        lambda_max = max(
            abs(self.WL[1]) + self.isentropic.sound_speed(self.WL),
            abs(self.WR[1]) + self.isentropic.sound_speed(self.WR)
        )
        FL = self.isentropic.compute_flux(self.UL, self.WL)
        FR = self.isentropic.compute_flux(self.UR, self.WR)
        F_LF = 0.5 * (FL + FR - lambda_max * (self.UR - self.UL))
        U_mid = 0.5 * (self.UL + self.UR) - 0.5 * (FR - FL) / (lambda_max + 1e-10)
        W_mid = self.isentropic.to_primitive(U_mid)
        F_Richtmyer = self.isentropic.compute_flux(U_mid, W_mid)
        expected = 0.5 * (F_LF + F_Richtmyer)
        np.testing.assert_array_almost_equal(F, expected, decimal=5)

    def test_hll_flux(self):
        """Test HLL flux computation."""
        flux = Flux(self.isentropic)
        F = flux.hll(self.UL, self.UR, self.WL, self.WR)
        cL = self.isentropic.sound_speed(self.WL)
        cR = self.isentropic.sound_speed(self.WR)
        SL = min(self.WL[1] - cL, self.WR[1] - cR)
        SR = max(self.WL[1] + cL, self.WR[1] + cR)
        FL = self.isentropic.compute_flux(self.UL, self.WL)
        FR = self.isentropic.compute_flux(self.UR, self.WR)
        expected = (SR * FL - SL * FR + SL * SR * (self.UR - self.UL)) / (SR - SL + 1e-10)
        np.testing.assert_array_almost_equal(F, expected, decimal=5)

    def test_hllc_flux(self):
        """Test HLLC flux computation."""
        flux = Flux(self.isentropic)
        F = flux.hllc(self.UL, self.UR, self.WL, self.WR)
        S_L, S_R, S_star = self.isentropic.hllc_wave_speeds(self.WL, self.WR, self.UL, self.UR)
        UL_star, UR_star = self.isentropic.hllc_intermediate_states(self.WL, self.WR, self.UL, self.UR, S_L, S_R, S_star)
        FL = self.isentropic.compute_flux(self.UL, self.WL)
        FR = self.isentropic.compute_flux(self.UR, self.WR)
        if S_L >= 0:
            expected = FL
        elif S_L <= 0 <= S_star:
            expected = FL + S_L * (UL_star - self.UL)
        elif S_star <= 0 <= S_R:
            expected = FR + S_R * (UR_star - self.UR)
        else:
            expected = FR
        np.testing.assert_array_almost_equal(F, expected, decimal=5)

    def test_roe_flux(self):
        """Test Roe flux computation."""
        flux = Flux(self.isentropic)
        F = flux.roe(self.UL, self.UR, self.WL, self.WR)
        eigenvalues, eigenvectors, delta = self.isentropic.roe_eigenstructure(self.WL, self.WR, self.UL, self.UR)
        alpha = self.isentropic.roe_wave_strengths(self.WL, self.WR, self.UL, self.UR)
        FL = self.isentropic.compute_flux(self.UL, self.WL)
        FR = self.isentropic.compute_flux(self.UR, self.WR)
        dissipative_term = np.zeros_like(self.UL)
        for i in range(len(eigenvalues)):
            lambda_i = eigenvalues[i]
            lambda_i = lambda_i if abs(lambda_i) > delta else 0.5 * (lambda_i + np.sqrt(lambda_i**2 + delta**2))
            dissipative_term += abs(lambda_i) * alpha[i] * eigenvectors[i]
        expected = 0.5 * (FL + FR - dissipative_term)
        np.testing.assert_array_almost_equal(F, expected, decimal=5)


if __name__ == '__main__':
    unittest.main()
