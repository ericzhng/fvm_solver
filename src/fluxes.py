import numpy as np
from .equation import EquationSystem, ShallowWaterSystem, EulerEquationSystem

class Flux:
    """Computes numerical fluxes for hyperbolic conservation laws."""
    
    def __init__(self, equation_system: EquationSystem, lambda_max: float = None):
        """Initialize flux computation.

        Args:
            equation_system: Equation system instance
            lambda_max: Global maximum wave speed for Lax-Friedrichs (optional)
        """
        self.equation_system = equation_system
        self.min_var = equation_system.min_var
        self.velocity_index = equation_system.velocity_index
        self.safeguarded_indices = equation_system.safeguarded_indices
        self.lambda_max = lambda_max or 1.0  # Default if not provided

    def _is_invalid_state(self, UL: np.ndarray, UR: np.ndarray) -> bool:
        """Check if conservative states violate safeguarded variable thresholds.
        
        Args:
            UL, UR: Left and right conservative states
            
        Returns:
            True if any safeguarded conservative variable is below threshold
        """
        for idx in self.safeguarded_indices:
            if UL[idx] <= self.min_var or UR[idx] <= self.min_var:
                return True
        return False

    def _handle_invalid_state(self, W_L: np.ndarray, W_R: np.ndarray, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Handle invalid states by selecting upwind flux.
        
        Args:
            W_L, W_R: Left and right primitive states
            U_L, U_R: Left and right conservative states
            
        Returns:
            Upwind flux
        """
        u_avg = 0.5 * (W_L[self.velocity_index] + W_R[self.velocity_index])
        return self.equation_system.compute_flux(
            U_L if u_avg >= 0 else U_R, 
            W_L if u_avg >= 0 else W_R
        )

    def lax_friedrichs(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Lax-Friedrichs flux: Simple and diffusive."""
        if self._is_invalid_state(UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)
        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)
        return 0.5 * (FL + FR - self.lambda_max * (UR - UL))

    def rusanov(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Rusanov flux: Local Lax-Friedrichs, less diffusive."""
        if self._is_invalid_state(UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)
        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)
        lambda_local = max(
            abs(WL[self.velocity_index]) + self.equation_system.sound_speed(WL),
            abs(WR[self.velocity_index]) + self.equation_system.sound_speed(WR)
        )
        return 0.5 * (FL + FR - lambda_local * (UR - UL))

    def force(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """FORCE flux: First-Order Centered, Combines Lax-Friedrichs and Richtmyer."""
        if self._is_invalid_state(UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)
        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)
        lambda_max = max(
            abs(WL[self.velocity_index]) + self.equation_system.sound_speed(WL),
            abs(WR[self.velocity_index]) + self.equation_system.sound_speed(WR)
        )
        F_LF = 0.5 * (FL + FR - lambda_max * (UR - UL))
        U_mid = 0.5 * (UL + UR) - 0.5 * (FR - FL) / (lambda_max + 1e-10)
        W_mid = self.equation_system.to_primitive(U_mid)
        F_Richtmyer = self.equation_system.compute_flux(U_mid, W_mid)
        return 0.5 * (F_LF + F_Richtmyer)

    def hll(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """HLL flux: Harten-Lax-van Leer, Uses two-wave approximation."""
        if self._is_invalid_state(UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)
        cL = self.equation_system.sound_speed(WL)
        cR = self.equation_system.sound_speed(WR)
        SL = min(WL[self.velocity_index] - cL, WR[self.velocity_index] - cR)
        SR = max(WL[self.velocity_index] + cL, WR[self.velocity_index] + cR)
        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)
        if SL >= 0:
            return FL
        if SR <= 0:
            return FR
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + 1e-10)

    def hllc(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """HLLC flux: Harten-Lax-van Leer-Contact, Includes contact wave for better resolution."""
        if self._is_invalid_state(UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)

        # Compute wave speeds
        S_L, S_R, S_star = self.equation_system.hllc_wave_speeds(WL, WR, UL, UR)
        
        # Compute intermediate states
        UL_star, UR_star = self.equation_system.hllc_intermediate_states(WL, WR, UL, UR, S_L, S_R, S_star)
        
        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)

        if S_L >= 0:
            return FL
        if S_L <= 0 <= S_star:
            return FL + S_L * (UL_star - UL)
        if S_star <= 0 <= S_R:
            return FR + S_R * (UR_star - UR)
        return FR

    def roe(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Roe flux: Generalized linearized Riemann solver with entropy fix."""
        if self._is_invalid_state(UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)
        
        # Compute Roe components
        eigenvalues, eigenvectors, delta = self.equation_system.roe_eigenstructure(WL, WR, UL, UR)
        alpha = self.equation_system.roe_wave_strengths(WL, WR, UL, UR)
        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)
        
        # Apply entropy fix and compute dissipative term
        dissipative_term = np.zeros_like(UL)
        for i in range(len(eigenvalues)):
            lambda_i = eigenvalues[i]
            lambda_i = lambda_i if abs(lambda_i) > delta else 0.5 * (lambda_i + np.sqrt(lambda_i**2 + delta**2))
            dissipative_term += abs(lambda_i) * alpha[i] * eigenvectors[i]
        
        return 0.5 * (FL + FR - dissipative_term)

    def get_flux(self, name: str):
        """Return flux function by name."""
        flux_methods = {
            'Lax-Friedrichs': self.lax_friedrichs,
            'Rusanov': self.rusanov,
            'FORCE': self.force,
            'HLL': self.hll,
            'HLLC': self.hllc,
            'Roe': self.roe
        }
        if name not in flux_methods:
            raise ValueError(f"Unsupported flux: {name}")
        return flux_methods[name]
