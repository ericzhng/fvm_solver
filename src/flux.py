import numpy as np
from .equation.base_equation import EquationSystem


def numerical_jacobian(F, U, h=1e-6):
    n = len(U)
    A = np.zeros((n, n))
    for j in range(n):
        U_plus = U.copy(); U_plus[j] += h
        U_minus = U.copy(); U_minus[j] -= h
        A[:, j] = (F(U_plus) - F(U_minus)) / (2 * h)
    return A


class Flux:
    """Class to compute numerical fluxes for hyperbolic conservation laws.

    Supports multiple flux methods: Lax-Friedrichs, Rusanov, FORCE, HLL, HLLC, Roe.
    """

    def __init__(self, equation_system: EquationSystem, lambda_max: float = 1.0):
        """Initialize the flux calculator.

        Args:
            equation_system (EquationSystem): The equation system to compute fluxes for.
            lambda_max (float): Maximum wave speed for Lax-Friedrichs (default: 1.0).
        """
        self.equation_system = equation_system
        self.min_var = equation_system.min_var
        self.velocity_index = equation_system.velocity_index
        self.safeguarded_indices = equation_system.safeguarded_indices
        self.lambda_max = lambda_max or 1.0

    def _is_invalid_state(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> bool:
        """Check if primitive states violate safeguarded variable thresholds.
        
        Args:
            WL, WR: Left and right primitive states
            
        Returns:
            True if any safeguarded primitive variable is below threshold
        """
        for idx in self.safeguarded_indices:
            if WL[idx] <= self.min_var or WR[idx] <= self.min_var:
                return True
        for U in [UL, UR]:
            W = self.equation_system.to_primitive(U)
            for idx in self.safeguarded_indices:
                if W[idx] <= self.min_var:
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
        """Compute Lax-Friedrichs flux.

        F = 0.5 * (FL + FR - lambda_max * (UR - UL))

        Args:
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray: Numerical flux.
        """
        if self._is_invalid_state(WL, WR, UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)
        
        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)
        return 0.5 * (FL + FR - self.lambda_max * (UR - UL))

    def rusanov(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Compute Rusanov (local Lax-Friedrichs) flux.

        F = 0.5 * (FL + FR - lambda_local * (UR - UL))

        Args:
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray: Numerical flux.
        """
        if self._is_invalid_state(WL, WR, UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)

        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)
        # Local maximum wave speed
        lambda_local = max(
            abs(WL[self.velocity_index]) + self.equation_system.sound_speed(WL),
            abs(WR[self.velocity_index]) + self.equation_system.sound_speed(WR)
        ) if self.velocity_index is not None else self.lambda_max

        return 0.5 * (FL + FR - lambda_local * (UR - UL))

    def force(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Compute FORCE flux (average of Lax-Friedrichs and Richtmyer).
        First-Order Centered

        F = 0.5 * (F_LF + F_Richtmyer)

        Args:
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray: Numerical flux.
        """
        if self._is_invalid_state(WL, WR, UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)

        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)

        # Local maximum wave speed
        lambda_local = max(
            abs(WL[self.velocity_index]) + self.equation_system.sound_speed(WL),
            abs(WR[self.velocity_index]) + self.equation_system.sound_speed(WR)
        ) if self.velocity_index is not None else self.lambda_max
        # Lax-Friedrichs component
        F_LF = 0.5 * (FL + FR - lambda_local * (UR - UL))

        # Richtmyer component
        U_mid = 0.5 * (UL + UR) - 0.5 * (FR - FL) / (lambda_local + self.min_var)
        W_mid = self.equation_system.to_primitive(U_mid)
        F_Richtmyer = self.equation_system.compute_flux(U_mid, W_mid)

        return 0.5 * (F_LF + F_Richtmyer)

    def hll(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Compute HLL flux.

        F = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)

        Args:
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray: Numerical flux.
        """
        if self._is_invalid_state(WL, WR, UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)

        cL = self.equation_system.sound_speed(WL)
        cR = self.equation_system.sound_speed(WR)
        uL = WL[self.equation_system.velocity_index] if self.equation_system.velocity_index is not None else 0.0
        uR = WR[self.equation_system.velocity_index] if self.equation_system.velocity_index is not None else 0.0
        
        # Wave speed estimates
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)

        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)

        if SL >= 0:
            return FL
        if SR <= 0:
            return FR
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + self.min_var)

    def hllc(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Compute HLLC flux with intermediate states.

        Args:
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray: Numerical flux.
        """
        if self._is_invalid_state(WL, WR, UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)
        
        # Use equation system's hllc_states_and_flux
        F = self.equation_system.hllc_numerical_flux(WL, WR, UL, UR)
        return F

    def roe(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """Compute Roe flux with entropy fix.

        F = 0.5 * (FL + FR - |A| * (UR - UL))

        Args:
            UL (np.ndarray): Left conservative state.
            UR (np.ndarray): Right conservative state.
            WL (np.ndarray): Left primitive state.
            WR (np.ndarray): Right primitive state.

        Returns:
            np.ndarray: Numerical flux.
        """
        if self._is_invalid_state(WL, WR, UL, UR):
            return self._handle_invalid_state(WL, WR, UL, UR)
        
        # Use equation system's roe_states_and_flux
        F = self.equation_system.roe_numerical_flux(WL, WR, UL, UR)
        return F

    def get_flux(self, flux_type: str):
        """Return the specified flux method.

        Args:
            flux_type (str): Type of flux ('lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe').

        Returns:
            callable: The flux computation function.

        Raises:
            ValueError: If flux_type is not supported.
        """
        flux_methods = {
            'lax_friedrichs': self.lax_friedrichs,
            'rusanov': self.rusanov,
            'force': self.force,
            'hll': self.hll,
            'hllc': self.hllc,
            'roe': self.roe
        }
        if flux_type not in flux_methods:
            raise ValueError(f"Unsupported flux type: {flux_type}. Choose from {list(flux_methods.keys())}")
        return flux_methods[flux_type]
