import numpy as np
from .equation.base_equation import EquationSystem


class Flux:
    """Compute numerical fluxes for hyperbolic conservation laws.

    Supports: Lax-Friedrichs, Rusanov, FORCE, HLL, HLLC, Roe.
    """

    def __init__(self, eqn_obj: EquationSystem, str_flux: str, lambda_max: float = 1.0):
        """
        Args:
            eqn_obj (EquationSystem): The equation system to compute fluxes for.
            lambda_max (float): Maximum wave speed for Lax-Friedrichs (default: 1.0).
        """
        self.eqn_obj = eqn_obj
        self.min_value = eqn_obj.min_value
        self.vel_idx = eqn_obj.vel_idx
        self.lambda_max = lambda_max or 1.0
        self.name = str_flux

        self.flux_dicts = {
            "lax": self.LF,  # Lax-Friedrichs
            "rusanov": self.RUS,
            "force": self.FORCE,
            "ausm": self.AUSM,
            "hll": self.HLL,
            "hlle": self.HLLE,
            "hllc": self.HLLC,
            "roe": self.ROE,
        }

        if self.name not in self.flux_dicts:
            raise ValueError(
                f"Unsupported flux type: {self.name}. Choose from {list(self.flux_dicts.keys())}"
            )

    def flux_func(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Return the specified flux method.

        Args:
            flux_type (str): Type of flux ('lax_friedrichs', 'rusanov', 'force', 'hll', 'hllc', 'roe').

        Returns:
            callable: The flux computation function.

        Raises:
            ValueError: If flux_type is not supported.
        """
        return self.flux_dicts[self.name](U_L, U_R)

    def LF(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Compute Lax-Friedrichs flux.

        F = 0.5 * (FL + FR - lambda_max * (U_R - U_L))

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Numerical flux.
        """
        FL = self.eqn_obj.compute_flux(U_L)
        FR = self.eqn_obj.compute_flux(U_R)

        return 0.5 * (FL + FR - self.lambda_max * (U_R - U_L))

    def RUS(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Compute Rusanov flux.

        F = 0.5 * (FL + FR - smax * (U_R - U_L))

            Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Numerical flux.
        """
        FL = self.eqn_obj.compute_flux(U_L)
        FR = self.eqn_obj.compute_flux(U_R)

        # Local maximum wave speed
        a_roe, u_roe = self.eqn_obj.roe_average(U_L, U_R)
        smax = abs(u_roe) + a_roe

        return 0.5 * (FL + FR - smax * (U_R - U_L))

    def FORCE(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Compute FORCE flux (average of Lax-Friedrichs and Richtmyer).

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Numerical flux.
        """
        FL = self.eqn_obj.compute_flux(U_L)
        FR = self.eqn_obj.compute_flux(U_R)

        # Local maximum wave speed
        a_roe, u_roe = self.eqn_obj.roe_average(U_L, U_R)
        smax = abs(u_roe) + a_roe

        # uL = self.eqn_obj.to_primitive(U_L)[self.vel_idx]
        # uR = self.eqn_obj.to_primitive(U_R)[self.vel_idx]
        # smax = max(
        #     uL + self.eqn_obj.sound_speed(U_L),
        #     uR + self.eqn_obj.sound_speed(U_R),
        # )

        # Lax-Friedrichs component
        F_LF = 0.5 * (FL + FR - smax * (U_R - U_L))

        # Richtmyer component
        U_mid = 0.5 * (U_L + U_R) - 0.5 * (FR - FL) / max(smax, self.min_value)
        F_Richtmyer = self.eqn_obj.compute_flux(U_mid)

        return 0.5 * (F_LF + F_Richtmyer)

    def HLL(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Compute HLL flux.

        F = (SR * FL - SL * FR + SL * SR * (U_R - U_L)) / (SR - SL)

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Numerical flux.
        """
        uL = self.eqn_obj.to_primitive(U_L)[self.vel_idx]
        uR = self.eqn_obj.to_primitive(U_R)[self.vel_idx]

        cL = self.eqn_obj.sound_speed(U_L)
        cR = self.eqn_obj.sound_speed(U_R)

        # Wave speed estimates
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)

        FL = self.eqn_obj.compute_flux(U_L)
        FR = self.eqn_obj.compute_flux(U_R)

        if SL >= 0:
            HLL = FL
        if SR <= 0:
            HLL = FR

        HLL = (SR * FL - SL * FR + SL * SR * (U_R - U_L)) / max(SR - SL, self.min_value)
        return HLL

    def AUSM(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Compute AUSM flux.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Numerical flux.
        """
        return self.eqn_obj.ausm_flux(U_L, U_R)

    def HLLC(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Compute HLLC flux.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Numerical flux.
        """
        return self.eqn_obj.hllc_flux(U_L, U_R)

    def HLLE(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Compute HLLC flux.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Numerical flux.
        """
        return self.eqn_obj.hlle_flux(U_L, U_R)

    def ROE(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Compute Roe flux with entropy fix.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: Numerical flux.
        """
        return self.eqn_obj.roe_flux(U_L, U_R)
