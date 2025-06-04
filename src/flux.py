import numpy as np
from .equation.base_equation import EquationSystem


def numerical_jacobian(F, U, h=1e-6):
    """Compute the numerical Jacobian matrix of F at U using central differences.

    Args:
        F (callable): Function mapping U to F(U).
        U (np.ndarray): State vector.
        h (float): Perturbation size.

    Returns:
        np.ndarray: Jacobian matrix.
    """
    n = len(U)
    A = np.zeros((n, n))
    for j in range(n):
        U_plus = U.copy(); U_plus[j] += h
        U_minus = U.copy(); U_minus[j] -= h
        A[:, j] = (F(U_plus) - F(U_minus)) / (2 * h)
    return A


class Flux:
    """Compute numerical fluxes for hyperbolic conservation laws.

    Supports: Lax-Friedrichs, Rusanov, FORCE, HLL, HLLC, Roe.
    """

    def __init__(self, equation_system: EquationSystem, lambda_max: float = 1.0):
        """

        Args:
            equation_system (EquationSystem): The equation system to compute fluxes for.
            lambda_max (float): Maximum wave speed for Lax-Friedrichs (default: 1.0).
        """
        self.equation_system = equation_system
        self.min_value = equation_system.min_value
        self.vel_idx = equation_system.vel_idx
        self.lambda_max = lambda_max or 1.0

    def _is_invalid_state(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> bool:
        """
        Check if primitive states violate positivity for safety-guarded variables.

        Args:
            WL, WR: Left and right primitive states
            UL, UR: Left and right conservative states

        Returns:
            bool: True if any safety-guarded primitive variable is below threshold.
        """
        # Use equation_system.safety_guard_var_idx for safety checks
        idxs = getattr(self.equation_system, "safety_guard_var_idx", [0])
        for idx in idxs:
            if WL[idx] <= self.min_value or WR[idx] <= self.min_value:
                return True
        for U in [UL, UR]:
            W = self.equation_system.to_primitive(U)
            for idx in idxs:
                if W[idx] <= self.min_value:
                    return True
        return False

    def _handle_invalid_state(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """
        Handle invalid states by selecting upwind flux.

        Args:
            WL, WR: Left and right primitive states
            UL, UR: Left and right conservative states

        Returns:
            np.ndarray: Upwind flux.
        """
        u_avg = 0.5 * (WL[self.vel_idx] + WR[self.vel_idx])
        return self.equation_system.compute_flux(
            UL if u_avg >= 0 else UR,
            WL if u_avg >= 0 else WR
        )

    def lax_friedrichs(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """
        Compute Lax-Friedrichs flux.

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
        """
        Compute Rusanov (local Lax-Friedrichs) flux.

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
            abs(WL[self.vel_idx]) + self.equation_system.sound_speed(WL),
            abs(WR[self.vel_idx]) + self.equation_system.sound_speed(WR)
        ) if self.vel_idx is not None else self.lambda_max

        return 0.5 * (FL + FR - lambda_local * (UR - UL))

    def force(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """
        Compute FORCE flux (average of Lax-Friedrichs and Richtmyer).

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

        lambda_local = max(
            abs(WL[self.vel_idx]) + self.equation_system.sound_speed(WL),
            abs(WR[self.vel_idx]) + self.equation_system.sound_speed(WR)
        ) if self.vel_idx is not None else self.lambda_max

        # Lax-Friedrichs component
        F_LF = 0.5 * (FL + FR - lambda_local * (UR - UL))

        # Richtmyer component
        U_mid = 0.5 * (UL + UR) - 0.5 * (FR - FL) / (lambda_local + self.min_value)
        W_mid = self.equation_system.to_primitive(U_mid)
        F_Richtmyer = self.equation_system.compute_flux(U_mid, W_mid)

        return 0.5 * (F_LF + F_Richtmyer)

    def hll(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """
        Compute HLL flux.

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
        uL = WL[self.vel_idx] if self.vel_idx is not None else 0.0
        uR = WR[self.vel_idx] if self.vel_idx is not None else 0.0

        # Wave speed estimates
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)

        FL = self.equation_system.compute_flux(UL, WL)
        FR = self.equation_system.compute_flux(UR, WR)

        if SL >= 0:
            return FL
        if SR <= 0:
            return FR
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + self.min_value)

    def hllc(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """
        Compute HLLC flux with intermediate states.

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

        return self.equation_system.hllc_numerical_flux(WL, WR, UL, UR)

    def roe(self, UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        """
        Compute Roe flux with entropy fix.

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

        return self.equation_system.roe_numerical_flux(WL, WR, UL, UR)

    def get_flux(self, flux_type: str):
        """
        Return the specified flux method.

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
