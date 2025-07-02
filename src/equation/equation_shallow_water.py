import numpy as np
from .equation_base import EquationBase


class ShallowWater(EquationBase):
    """
    1D Shallow Water Equation System.

    Governs the conservation of water height and momentum, including gravitational effects.
    Primitive variables: [height, velocity]
    Conservative variables: [height, momentum]
    """

    def __init__(self, gravity: float = 9.81):
        """
        Initialize the shallow water equation system.

        Args:
            gravity (float): Gravitational acceleration (must be positive).
        """
        if gravity <= 0:
            raise ValueError("gravity must be positive")
        super().__init__(min_value=1e-10)
        self.g = gravity
        self.var_names = ["height", "velocity"]
        self.num_vars = len(self.var_names)
        self.vel_idx = 1

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """
        Convert primitive variables to conservative variables.

        Args:
            W (np.ndarray): Primitive variables [height, velocity], shape (2,).

        Returns:
            np.ndarray: Conservative variables [height, momentum], shape (2,).
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        h, u = W
        h = np.maximum(h, self.min_value)
        return np.array([h, h * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """
        Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables [height, momentum], shape (2,).

        Returns:
            np.ndarray: Primitive variables [height, velocity], shape (2,).
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        h, hu = U
        h = np.maximum(h, self.min_value)
        return np.array([h, hu / h])

    def sound_speed(self, U: np.ndarray) -> float:
        """
        Compute the local wave speed (gravity wave speed).

        Args:
            W (np.ndarray): Primitive variables [height, velocity], shape (2,).

        Returns:
            float: Local wave speed.
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        h = U[0]
        h = np.maximum(h, self.min_value)
        return np.sqrt(self.g * h)

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """Compute the physical flux vector.

        Args:
            U (np.ndarray): Conservative variables [height, momentum], shape (2,).
            W (np.ndarray): Primitive variables [height, velocity], shape (2,).

        Returns:
            np.ndarray: Flux vector, shape (2,).
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        h, hu = U
        h = np.maximum(h, self.min_value)
        u = hu / h
        return np.array([h * u, h * u**2 + 0.5 * self.g * h**2])

    def roe_average(self, U_L: np.ndarray, U_R: np.ndarray) -> tuple:
        """Compute the Roe averages for the shallow water equations, with entropy fix.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum, energy]
            U_R (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            tuple(np.ndarray): Roe averages
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        hL, huL = self.to_primitive(U_L)
        hL = np.maximum(hL, self.min_value)
        uL = huL / hL

        # right state
        hR, huR = self.to_primitive(U_R)
        hR = np.maximum(hR, self.min_value)
        uR = huR / hR

        # Roe averages
        sqrt_hL = np.sqrt(hL)
        sqrt_hR = np.sqrt(hR)

        u_roe = (uL * sqrt_hL + uR * sqrt_hR) / np.maximum(
            sqrt_hL + sqrt_hR, self.min_value
        )

        h_roe = sqrt_hL * sqrt_hR
        h_roe = np.maximum(h_roe, self.min_value)

        a_roe = np.sqrt(self.g * h_roe)

        return a_roe, u_roe

    # ---------------------------------------------------- #
    # flux methods that has to be defined per equation wise
    # ---------------------------------------------------- #

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
        hL, huL = self.to_primitive(U_L)
        hL = np.maximum(hL, self.min_value)
        uL = huL / hL
        aL = np.sqrt(self.g * hL)
        ML = uL / aL  # Mach

        # right state
        hR, huR = self.to_primitive(U_R)
        hR = np.maximum(hR, self.min_value)
        uR = huR / hR
        aR = np.sqrt(self.g * hR)
        MR = uR / aR

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
        """Compute the HLLC numerical flux for the shallow water equations.

        Args:
            W_L (np.ndarray): Left primitive state [height, velocity], shape (2,).
            W_R (np.ndarray): Right primitive state [height, velocity], shape (2,).
            U_L (np.ndarray): Left conservative state [height, momentum], shape (2,).
            U_R (np.ndarray): Right conservative state [height, momentum], shape (2,).

        Returns:
            np.ndarray: HLLC numerical flux, shape (2,).
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)
        hL, uL = W_L
        hR, uR = W_R
        hL = np.maximum(hL, self.min_value)
        hR = np.maximum(hR, self.min_value)

        cL = np.sqrt(self.g * hL)
        cR = np.sqrt(self.g * hR)
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)
        denom = hR * (SR - uR) - hL * (SL - uL)
        S_star = (
            hR * uR * (SR - uR) - hL * uL * (SL - uL) + 0.5 * self.g * (hR**2 - hL**2)
        ) / (denom + self.min_value)

        hL_star = np.maximum(
            hL * (SL - uL) / (SL - S_star + self.min_value), self.min_value
        )
        hR_star = np.maximum(
            hR * (SR - uR) / (SR - S_star + self.min_value), self.min_value
        )
        UL_star = np.array([hL_star, hL_star * S_star])
        UR_star = np.array([hR_star, hR_star * S_star])

        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if SL >= 0:
            F = FL
        elif SL <= 0 <= S_star:
            F = FL + SL * (UL_star - U_L)
        elif S_star <= 0 <= SR:
            F = FR + SR * (UR_star - U_R)
        else:
            F = FR

        return F

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute the Roe flux for the shallow water equations, with entropy fix.

        Args:
            W_L (np.ndarray): Left primitive state [height, velocity], shape (2,).
            W_R (np.ndarray): Right primitive state [height, velocity], shape (2,).
            U_L (np.ndarray): Left conservative state [height, momentum], shape (2,).
            U_R (np.ndarray): Right conservative state [height, momentum], shape (2,).

        Returns:
            np.ndarray: Roe numerical flux, shape (2,).
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        hL, huL = self.to_primitive(U_L)
        hL = np.maximum(hL, self.min_value)
        uL = huL / hL
        aL = np.sqrt(self.g * hL)

        # right state
        hR, huR = self.to_primitive(U_R)
        hR = np.maximum(hR, self.min_value)
        uR = huR / hR
        aR = np.sqrt(self.g * hR)

        # Roe averages
        sqrt_hL = np.sqrt(hL)
        sqrt_hR = np.sqrt(hR)
        u_roe = (uL * sqrt_hL + uR * sqrt_hR) / np.maximum(
            sqrt_hL + sqrt_hR, self.min_value
        )
        h_roe = sqrt_hL * sqrt_hR
        h_roe = np.maximum(h_roe, self.min_value)
        c_roe = np.sqrt(self.g * h_roe)

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # Differences in primitive variables
        dh = hR - hL
        dhu = huR - huR

        # Wave strength (Characteristic Variables)
        dV = np.array(
            [
                (dhu - u_roe * dh + c_roe * dh) / (2 * c_roe),
                (dhu - u_roe * dh - c_roe * dh) / (-2 * c_roe),
            ]
        )

        # Absolute values of the wave speeds (Eigenvalues)
        ws = np.array([abs(u_roe - c_roe), abs(u_roe + c_roe)])

        # Harten's Entropy Fix JCP(1983), 49, pp357-393
        Da = max(0, 4 * ((uR - aR) - (uL - aL)))
        if ws[0] < Da / 2 and Da != 0:
            ws[0] = ws[0] ** 2 / Da + Da / 4

        Da = max(0, 4 * ((uR + aR) - (uL + aL)))
        if ws[1] < Da / 2 and Da != 0:
            ws[1] = ws[1] ** 2 / Da + Da / 4

        # Right eigenvectors
        R = np.array(
            [
                [1, 1],
                [u_roe + c_roe, u_roe - c_roe],
            ]
        )

        # Add the matrix dissipation term to complete the Roe flux
        F = (FL + FR - R @ (ws * dV)) / 2.0

        return F
