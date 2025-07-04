"""
This module defines the Flux class, which provides a unified interface for various
numerical flux schemes used in the FVM solver.
"""

import numpy as np
from .equation.equation_base import EqnBase


class Flux:
    """
    Computes numerical fluxes for hyperbolic conservation laws.

    This class acts as a factory for different numerical flux functions.
    It takes an equation object and a flux type string, and provides a
    callable method to compute the requested numerical flux at cell interfaces.

    Supported flux schemes:
    - 'lax': Lax-Friedrichs
    - 'rusanov': Rusanov (or local Lax-Friedrichs)
    - 'force': First-Order Centered (FORCE) scheme
    - 'ausm': Advection Upstream Splitting Method (delegated to equation class)
    - 'hll': Harten-Lax-van Leer
    - 'hlle': Harten-Lax-van Leer-Einfeldt (delegated to equation class)
    - 'hllc': Harten-Lax-van Leer-Contact (delegated to equation class)
    - 'roe': Roe's approximate Riemann solver (delegated to equation class)
    """

    def __init__(self, eqn_obj: EqnBase, str_flux: str, lambda_max: float = 1.0):
        """
        Initializes the Flux computer.

        Args:
            eqn_obj (EqnBase): The equation system object (e.g., EqnEuler) for
                                which fluxes will be computed.
            str_flux (str): The name of the numerical flux scheme to use.
            lambda_max (float, optional): A global maximum wave speed, used by
                                         the Lax-Friedrichs scheme. Defaults to 1.0.
        """
        self.eqn_obj = eqn_obj
        self.min_value = eqn_obj.min_value
        self.vel_idx = eqn_obj.vel_idx
        self.lambda_max = lambda_max
        self.name = str_flux.lower()

        # Dictionary mapping flux names to their corresponding methods
        self.flux_dicts = {
            "lax": self.lax_friedrichs,
            "rusanov": self.rusanov,
            "force": self.force,
            "ausm": self.ausm,
            "hll": self.hll,
            "hlle": self.hlle,
            "hllc": self.hllc,
            "roe": self.roe,
        }

        if self.name not in self.flux_dicts:
            valid_fluxes = list(self.flux_dicts.keys())
            raise ValueError(
                f"Unsupported flux type: '{self.name}'. " f"Choose from {valid_fluxes}"
            )

    def flux_func(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the numerical flux between two states using the selected scheme.

        Args:
            U_L (np.ndarray): The conservative state vector at the left of the interface.
            U_R (np.ndarray): The conservative state vector at the right of the interface.

        Returns:
            np.ndarray: The computed numerical flux vector.
        """
        return self.flux_dicts[self.name](U_L, U_R)

    def lax_friedrichs(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Lax-Friedrichs (LF) numerical flux.

        F_LF = 0.5 * (F(U_L) + F(U_R)) - 0.5 * λ_max * (U_R - U_L)

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: The Lax-Friedrichs numerical flux.
        """
        F_L = self.eqn_obj.compute_flux(U_L)
        F_R = self.eqn_obj.compute_flux(U_R)
        return 0.5 * (F_L + F_R - self.lambda_max * (U_R - U_L))

    def rusanov(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Rusanov (or local Lax-Friedrichs) numerical flux.

        F_Rusanov = 0.5 * (F(U_L) + F(U_R)) - 0.5 * s_max * (U_R - U_L)

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: The Rusanov numerical flux.
        """
        F_L = self.eqn_obj.compute_flux(U_L)
        F_R = self.eqn_obj.compute_flux(U_R)

        # Local maximum wave speed at the interface
        s_max = max(self.eqn_obj.max_eigenvalue(U_L), self.eqn_obj.max_eigenvalue(U_R))

        return 0.5 * (F_L + F_R - s_max * (U_R - U_L))

    def force(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the FORCE (First-Order Centered) numerical flux.
        This is an average of the Lax-Friedrichs and Richtmyer fluxes.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: The FORCE numerical flux.
        """
        F_L = self.eqn_obj.compute_flux(U_L)
        F_R = self.eqn_obj.compute_flux(U_R)

        # Local maximum wave speed
        s_max = max(self.eqn_obj.max_eigenvalue(U_L), self.eqn_obj.max_eigenvalue(U_R))

        # Lax-Friedrichs component
        F_LF = 0.5 * (F_L + F_R - s_max * (U_R - U_L))

        # Richtmyer component (two-step Lax-Wendroff)
        U_mid = 0.5 * (U_L + U_R) - 0.5 * (F_R - F_L) / max(s_max, self.min_value)
        F_Richtmyer = self.eqn_obj.compute_flux(U_mid)

        return 0.5 * (F_LF + F_Richtmyer)

    def hll(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Computes the Harten-Lax-van Leer (HLL) numerical flux.

        F_HLL = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)) / (S_R - S_L)

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: The HLL numerical flux.
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

    def ausm(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Delegates AUSM flux computation to the specific equation class.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: The AUSM numerical flux.
        """
        return self.eqn_obj.ausm_flux(U_L, U_R)

    def hllc(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Delegates HLLC flux computation to the specific equation class.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: The HLLC numerical flux.
        """
        return self.eqn_obj.hllc_flux(U_L, U_R)

    def hlle(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Delegates HLLE flux computation to the specific equation class.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: The HLLE numerical flux.
        """
        return self.eqn_obj.hlle_flux(U_L, U_R)

    def roe(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """
        Delegates Roe flux computation to the specific equation class.

        Args:
            U_L (np.ndarray): Left conservative state.
            U_R (np.ndarray): Right conservative state.

        Returns:
            np.ndarray: The Roe numerical flux.
        """
        return self.eqn_obj.roe_numerical_flux(U_L, U_R)
