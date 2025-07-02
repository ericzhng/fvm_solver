import numpy as np
from .equation_base import EqnBase


class EqnEuler(EqnBase):
    """1D Euler equations for compressible gas dynamics.

    Models conservation of mass, momentum, and energy.
    """

    def __init__(self, gamma: float = 1.4):
        """Initialize the Euler equation system.

        Args:
            gamma (float): Ratio of specific heats (default: 1.4 for air).
        """
        super().__init__(min_value=1e-10)
        self.gamma = gamma
        self.var_names = ["density", "velocity", "pressure"]
        self.num_vars = len(self.var_names)
        self.vel_idx = 1  # index start from 0, so velocity is at index 1

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Args:
            W (np.ndarray): [density, velocity, pressure]

        Returns:
            np.ndarray: [density, momentum, total energy]
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        rho, u, p = W
        rho = np.maximum(rho, self.min_value)
        p = np.maximum(p, self.min_value)
        E = p / (self.gamma - 1) + 0.5 * rho * u**2
        return np.array([rho, rho * u, E])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): [density, momentum, total energy]

        Returns:
            np.ndarray: [density, velocity, pressure]
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, m, E = U
        rho = np.maximum(rho, self.min_value)
        u = m / rho
        p = (E - 0.5 * rho * u**2) * (self.gamma - 1)
        p = np.maximum(p, self.min_value)
        return np.array([rho, u, p])

    def sound_speed(self, U: np.ndarray) -> float:
        """Compute the sound speed for the gas.

        Args:
            W (np.ndarray): [density, velocity, pressure]

        Returns:
            float: Sound speed (sqrt(gamma * p / rho))
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, m, E = U
        rho = np.maximum(rho, self.min_value)
        u = m / rho
        p = (E - 0.5 * rho * u**2) * (self.gamma - 1)
        p = np.maximum(p, self.min_value)
        return np.sqrt(self.gamma * p / rho)

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Args:
            U (np.ndarray): [density, momentum, total energy]
            W (np.ndarray): [density, velocity, pressure]

        Returns:
            np.ndarray: [rho*u, rho*u^2 + p, u*(E + p)]
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")

        rho, m, E = U
        rho = np.maximum(rho, self.min_value)
        u = m / rho
        p = (E - 0.5 * rho * u**2) * (self.gamma - 1)
        p = np.maximum(p, self.min_value)

        return np.array([rho * u, rho * u**2 + p, u * (E + p)])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute the Roe averages for the Euler equations.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum, energy]
            U_R (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            tuple(np.ndarray): Roe averages
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        rhoL, uL, pL = self.to_primitive(U_L)
        EL = U_L[2]

        # right state
        rhoR, uR, pR = self.to_primitive(U_R)
        ER = U_R[2]

        # Compute enthalpy
        hL = (EL + pL) / rhoL
        hR = (ER + pR) / rhoR

        # Roe averages
        sqrt_rhoL = np.sqrt(rhoL)
        sqrt_rhoR = np.sqrt(rhoR)
        u_roe = (uL * sqrt_rhoL + uR * sqrt_rhoR) / np.maximum(
            sqrt_rhoL + sqrt_rhoR, self.min_value
        )
        h_roe = (hL * sqrt_rhoL + hR * sqrt_rhoR) / np.maximum(
            sqrt_rhoL + sqrt_rhoR, self.min_value
        )
        a2 = (self.gamma - 1) * (h_roe - 0.5 * u_roe**2)
        a2 = np.maximum(a2, self.min_value)
        c_roe = np.sqrt(a2)

        return u_roe, c_roe

    # ---------------------------------------------------- #
    # flux methods that has to be defined per equation wise
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute the Roe flux for the Euler equations, with entropy fix.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum, energy]
            U_R (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            np.ndarray: Roe numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        rhoL, uL, pL = self.to_primitive(U_L)
        EL = U_L[2]
        aL = np.sqrt(self.gamma * pL / rhoL)
        HL = (EL + pL) / rhoL  # enthalpy

        # right state
        rhoR, uR, pR = self.to_primitive(U_R)
        ER = U_R[2]
        aR = np.sqrt(self.gamma * pR / rhoR)
        HR = (ER + pR) / rhoR

        # Roe averages
        sqrt_rhoL = np.sqrt(rhoL)
        sqrt_rhoR = np.sqrt(rhoR)
        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * sqrt_rhoL + uR * sqrt_rhoR) / np.maximum(
            sqrt_rhoL + sqrt_rhoR, self.min_value
        )
        H_roe = (HL * sqrt_rhoL + HR * sqrt_rhoR) / np.maximum(
            sqrt_rhoL + sqrt_rhoR, self.min_value
        )
        a2 = (self.gamma - 1) * (H_roe - 0.5 * u_roe**2)
        a2 = np.maximum(a2, self.min_value)
        c_roe = np.sqrt(a2)

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # Differences in primitive variables
        dr = rhoR - rhoL
        du = uR - uL
        dP = pR - pL

        # Wave strengths (characteristic variables)
        alphaMat = np.array(
            [
                (dP - rho_roe * c_roe * du) / (2 * c_roe**2),
                -(dP / (c_roe**2) - dr),
                (dP + rho_roe * c_roe * du) / (2 * c_roe**2),
            ]
        )

        # Absolute values of the wave speeds (Eigenvalues)
        lambdas = np.array([abs(u_roe - c_roe), abs(u_roe), abs(u_roe + c_roe)])

        # Harten's Entropy Fix JCP(1983), 49, pp357-393
        Da = max(0, 4 * ((uR - aR) - (uL - aL)))
        if lambdas[0] < Da / 2 and Da != 0:
            lambdas[0] = lambdas[0] ** 2 / Da + Da / 4

        Da = max(0, 4 * ((uR + aR) - (uL + aL)))
        if lambdas[2] < Da / 2 and Da != 0:
            lambdas[2] = lambdas[2] ** 2 / Da + Da / 4

        # Right eigenvectors
        R = np.array(
            [
                [1, 1, 1],
                [u_roe - c_roe, u_roe, u_roe + c_roe],
                [H_roe - u_roe * c_roe, u_roe**2 / 2, H_roe + u_roe * c_roe],
            ]
        )

        # Add the matrix dissipation term to complete the Roe flux
        Roe = (FL + FR - R @ (lambdas * alphaMat)) / 2

        return Roe

    def ausm_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum, energy]
            U_R (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            np.ndarray: AUSM numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        rhoL, uL, pL = self.to_primitive(U_L)
        EL = U_L[2]
        aL = np.sqrt(self.gamma * pL / rhoL)
        ML = uL / aL  # Mach
        HL = (EL + pL) / rhoL  # enthalpy

        # right state
        rhoR, uR, pR = self.to_primitive(U_R)
        ER = U_R[2]
        aR = np.sqrt(self.gamma * pR / rhoR)
        MR = uR / aR
        HR = (ER + pR) / rhoR

        # Positive M and p in the LEFT cell
        if ML <= -1:
            Mp = 0
            Pp = 0
        elif ML < 1:
            Mp = ((ML + 1) ** 2) / 4
            Pp = pL * ((1 + ML) ** 2) * (2 - ML) / 4  # or Pp = (1 + ML) * pL / 2
        else:
            Mp = ML
            Pp = pL

        # Negative M and p in the RIGHT cell
        if MR <= -1:
            Mm = MR
            Pm = pR
        elif MR < 1:
            Mm = -((MR - 1) ** 2) / 4
            Pm = pR * ((1 - MR) ** 2) * (2 + MR) / 4  # or Pm = (1 - MR) * pR / 2
        else:
            Mm = 0
            Pm = 0

        # Positive Part of Flux evaluated in the left cell
        MpMm = Mp + Mm
        Fp = np.zeros(3)
        Fp[0] = max(0, MpMm) * aL * rhoL
        Fp[1] = max(0, MpMm) * aL * rhoL * uL + Pp
        Fp[2] = max(0, MpMm) * aL * rhoL * HL

        # Negative Part of Flux evaluated in the right cell
        Fm = np.zeros(3)
        Fm[0] = min(0, MpMm) * aR * rhoR
        Fm[1] = min(0, MpMm) * aR * rhoR * uR + Pm
        Fm[2] = min(0, MpMm) * aR * rhoR * HR

        return Fp + Fm

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute HLLC numerical flux for Euler equations.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum, energy]
            U_R (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            np.ndarray: HLLC numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        rhoL, uL, pL = self.to_primitive(U_L)
        EL = U_L[2]
        aL = np.sqrt(self.gamma * pL / rhoL)

        # right state
        rhoR, uR, pR = self.to_primitive(U_R)
        ER = U_R[2]
        aR = np.sqrt(self.gamma * pR / rhoR)

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # Compute guess pressure from PVRS Riemann solver
        PPV = max(
            0, 0.5 * (pL + pR) + 0.5 * (uL - uR) * (0.25 * (rhoL + rhoR) * (aL + aR))
        )
        pmin = min(pL, pR)
        pmax = max(pL, pR)
        Qmax = pmax / pmin
        Quser = 2.0  # parameter manually set

        if (Qmax <= Quser) and (pmin <= PPV) and (PPV <= pmax):
            # Select PRVS Riemann solver
            pM = PPV
            # uM = 0.5 * (uL + uR) + 0.5 * (pL - pR) / CUP
        else:
            if PPV < pmin:
                # Select Two-Rarefaction Riemann solver
                PQ = (pL / pR) ** ((self.gamma - 1.0) / (2.0 * self.gamma))
                uM = (PQ * uL / aL + uR / aR + 2 / (self.gamma - 1) * (PQ - 1.0)) / (
                    PQ / aL + 1.0 / aR
                )
                PTL = 1 + (self.gamma - 1) / 2.0 * (uL - uM) / aL
                PTR = 1 + (self.gamma - 1) / 2.0 * (uM - uR) / aR
                pM = 0.5 * (
                    pL * PTL ** (2 * self.gamma / (self.gamma - 1))
                    + pR * PTR ** (2 * self.gamma / (self.gamma - 1))
                )
            else:
                # Use Two-Shock Riemann solver with PVRS as estimate
                GEL = np.sqrt(
                    (2 / (self.gamma + 1) / rhoL)
                    / ((self.gamma - 1) / (self.gamma + 1) * pL + PPV)
                )
                GER = np.sqrt(
                    (2 / (self.gamma + 1) / rhoR)
                    / ((self.gamma - 1) / (self.gamma + 1) * pR + PPV)
                )
                pM = (GEL * pL + GER * pR - (uR - uL)) / (GEL + GER)
                # uM = 0.5 * (uL + uR) + 0.5 * (GER * (pM - pR) - GEL * (pM - pL))

        # Estimate wave speeds: SL, SR and SM (Toro, 1994)
        zL = (
            np.sqrt(1 + (self.gamma + 1) / (2 * self.gamma) * (pM / pL - 1))
            if pM > pL
            else 1
        )
        zR = (
            np.sqrt(1 + (self.gamma + 1) / (2 * self.gamma) * (pM / pR - 1))
            if pM > pR
            else 1
        )

        SL = uL - aL * zL
        SR = uR + aR * zR
        SM = (pL - pR + rhoR * uR * (SR - uR) - rhoL * uL * (SL - uL)) / (
            rhoR * (SR - uR) - rhoL * (SL - uL)
        )

        # Compute the HLL flux.
        if 0 <= SL:
            HLLC = FL
        elif SL <= 0 <= SM:
            qsL = (
                rhoL
                * (SL - uL)
                / (SL - SM)
                * np.array(
                    [1, SM, EL / rhoL + (SM - uL) * (SM + pL / (rhoL * (SL - uL)))]
                )
            )
            HLLC = FL + SL * (qsL - U_L)
        elif SM <= 0 <= SR:
            qsR = (
                rhoR
                * (SR - uR)
                / (SR - SM)
                * np.array(
                    [1, SM, ER / rhoR + (SM - uR) * (SM + pR / (rhoR * (SR - uR)))]
                )
            )
            HLLC = FR + SR * (qsR - U_R)
        elif 0 >= SR:
            HLLC = FR

        return HLLC

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum, energy]
            U_R (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            np.ndarray: AUSM numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        rhoL, uL, pL = self.to_primitive(U_L)
        EL = U_L[2]
        aL = np.sqrt(self.gamma * pL / rhoL)
        HL = (EL + pL) / rhoL  # enthalpy

        # right state
        rhoR, uR, pR = self.to_primitive(U_R)
        ER = U_R[2]
        aR = np.sqrt(self.gamma * pR / rhoR)
        HR = (ER + pR) / rhoR

        # Roe averages
        sqrt_rhoL = np.sqrt(rhoL)
        sqrt_rhoR = np.sqrt(rhoR)
        u_roe = (uL * sqrt_rhoL + uR * sqrt_rhoR) / np.maximum(
            sqrt_rhoL + sqrt_rhoR, self.min_value
        )
        H_roe = (HL * sqrt_rhoL + HR * sqrt_rhoR) / np.maximum(
            sqrt_rhoL + sqrt_rhoR, self.min_value
        )
        a2 = (self.gamma - 1) * (H_roe - 0.5 * u_roe**2)
        a2 = np.maximum(a2, self.min_value)
        c_roe = np.sqrt(a2)

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # Wave speed estimates
        SLm = min(uL - aL, u_roe - c_roe)
        SRp = max(uR + aR, u_roe + c_roe)

        # Compute the HLL flux
        if SLm >= 0:  # Right-going supersonic flow
            HLLE = FL
        elif SLm <= 0 <= SRp:  # Subsonic flow
            select = 1
            if select == 1:
                # True HLLE function
                HLLE = (SRp * FL - SLm * FR + SLm * SRp * (U_R - U_L)) / (SRp - SLm)
            elif select == 2:
                # Rusanov flux (as suggested by Toro's book)
                smax = max(abs(SLm), abs(SRp))
                HLLE = (FR + FL + smax * (U_L - U_R)) / 2.0
        elif SRp <= 0:  # Left-going supersonic flow
            HLLE = FR

        return HLLE
