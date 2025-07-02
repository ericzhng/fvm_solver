import numpy as np
from .equation_base import EqnBase


class EqnIsentropicGas(EqnBase):
    """Isentropic gas equation system for 1D flows.

    Models conservation of mass and momentum under isentropic conditions.
    """

    def __init__(self, gamma: float = 1.4, k: float = 1.0):
        """Initialize the isentropic gas system.

        Args:
            gamma (float): Ratio of specific heats (defaU_Lt: 1.4).
            k (float): Gas constant (defaU_Lt: 1.0).
        """
        super().__init__(min_value=1e-10)
        self.gamma = gamma
        self.k = k
        self.var_names = ["density", "velocity"]
        self.num_vars = len(self.var_names)
        self.vel_idx = 1

    def to_conservative(self, W: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables.

        Primitive: [density, velocity]
        Conservative: [density, momentum = density * velocity]

        Args:
            W (np.ndarray): Primitive variables [density, velocity].

        RetU_Rns:
            np.ndarray: Conservative variables [density, momentum].
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        rho, u = W
        rho = np.maximum(rho, self.min_value)
        return np.array([rho, rho * u])

    def to_primitive(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables [density, momentum].

        RetU_Rns:
            np.ndarray: Primitive variables [density, velocity].
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, m = U
        rho = np.maximum(rho, self.min_value)
        return np.array([rho, m / rho])

    def sound_speed(self, U: np.ndarray) -> float:
        """Compute the sound speed for the isentropic gas.

        Args:
            W (np.ndarray): Primitive variables [density, velocity].

        RetU_Rns:
            float: Sound speed sqrt(gamma * k * rho^(gamma-1)).
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho = U[0]
        rho = np.maximum(rho, self.min_value)
        return np.sqrt(self.gamma * self.k * rho ** (self.gamma - 1))

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Flux: [rho*u, rho*u^2 + k*rho^gamma]

        Args:
            U (np.ndarray): Conservative variables [density, momentum].
            W (np.ndarray): Primitive variables [density, velocity].

        RetU_Rns:
            np.ndarray: Flux vector.
        """
        if U.shape != (self.num_vars,):
            raise ValueError(f"U must have shape ({self.num_vars},)")
        rho, m = U
        rho = np.maximum(rho, self.min_value)
        u = m / rho
        p = self.k * rho**self.gamma
        return np.array([rho * u, rho * u**2 + p])

    def roe_averages(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute Roe-averaged values for isentropic gas.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum].
            U_R (np.ndarray): Right conservative state [density, momentum].

        RetU_Rns:
            np.ndarray: Roe numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)

        rhoL, uL = W_L
        rhoR, uR = W_R
        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (
            np.sqrt(rhoL) + np.sqrt(rhoR) + self.min_value
        )
        rho_roe = np.maximum(rho_roe, self.min_value)
        c_roe = np.sqrt(self.gamma * self.k * rho_roe ** (self.gamma - 1))

        return u_roe, c_roe

    # ---------------------------------------------------- #
    # flux methods that has to be defined per equation wise
    # ---------------------------------------------------- #

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute Roe-averaged state, eigenstructU_Re, wave strengths, and flux for isentropic gas.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum].
            U_R (np.ndarray): Right conservative state [density, momentum].

        RetU_Rns:
            np.ndarray: Roe numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)
        rhoL, uL = W_L
        rhoR, uR = W_R
        aL = np.sqrt(self.gamma * self.k * rhoL ** (self.gamma - 1))
        aR = np.sqrt(self.gamma * self.k * rhoR ** (self.gamma - 1))

        rho_roe = np.sqrt(rhoL * rhoR)
        u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (
            np.sqrt(rhoL) + np.sqrt(rhoR) + self.min_value
        )
        rho_roe = np.maximum(rho_roe, self.min_value)
        c_roe = np.sqrt(self.gamma * self.k * rho_roe ** (self.gamma - 1))

        # Left and Right fluxes
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)

        # Differences in primitive variables
        dr = rhoR - rhoL
        dhu = U_R[1] - U_L[1]

        # Wave strengths (characteristic variables)
        alphaMat = np.array(
            [
                (dhu - u_roe * dr - c_roe * dr) / (-2 * c_roe),
                (dhu - u_roe * dr + c_roe * dr) / (2 * c_roe),
            ]
        )

        # Absolute values of the wave speeds (Eigenvalues)
        lambdas = np.array([abs(u_roe - c_roe), abs(u_roe + c_roe)])

        # Harten's Entropy Fix JCP(1983), 49, pp357-393
        Da = max(0, 4 * ((uR - aR) - (uL - aL)))
        if lambdas[0] < Da / 2 and Da != 0:
            lambdas[0] = lambdas[0] ** 2 / Da + Da / 4

        Da = max(0, 4 * ((uR + aR) - (uL + aL)))
        if lambdas[1] < Da / 2 and Da != 0:
            lambdas[1] = lambdas[1] ** 2 / Da + Da / 4

        # Right eigenvectors
        R = np.array(
            [
                [1, 1],
                [u_roe - c_roe, u_roe + c_roe],
            ]
        )

        # Add the matrix dissipation term to complete the Roe flux
        Roe = (FL + FR - R @ (lambdas * alphaMat)) / 2

        return Roe

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute HLLC wave speeds, intermediate states, and flux for isentropic gas.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum].
            U_R (np.ndarray): Right conservative state [density, momentum].

        RetU_Rns:
            np.ndarray: HLLC numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)

        # Extract variables
        rhoL, uL = W_L
        rhoR, uR = W_R
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)

        # Compute wave speeds
        cL = self.sound_speed(W_L)
        cR = self.sound_speed(W_R)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = (
            0.5 * (uL + uR)
            if abs(denom) < self.min_value
            else (pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / denom
        )

        # Compute intermediate states
        rho_star_L = max(
            rhoL * (S_L - uL) / (S_L - S_star + self.min_value), self.min_value
        )
        rho_star_R = max(
            rhoR * (S_R - uR) / (S_R - S_star + self.min_value), self.min_value
        )
        U_L_star = np.array([rho_star_L, rho_star_L * S_star])
        U_R_star = np.array([rho_star_R, rho_star_R * S_star])

        # Compute flux
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if S_L >= 0:
            F = FL
        elif S_L <= 0 <= S_star:
            F = FL + S_L * (U_L_star - U_L)
        elif S_star <= 0 <= S_R:
            F = FR + S_R * (U_R_star - U_R)
        else:
            F = FR

        return F

    def hlle_flux(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute HLLC wave speeds, intermediate states, and flux for isentropic gas.

        Args:
            U_L (np.ndarray): Left conservative state [density, momentum].
            U_R (np.ndarray): Right conservative state [density, momentum].

        RetU_Rns:
            np.ndarray: HLLC numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [U_L, U_R]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # left state
        W_L = self.to_primitive(U_L)
        W_R = self.to_primitive(U_R)

        # Extract variables
        rhoL, uL = W_L
        rhoR, uR = W_R
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)

        # Compute wave speeds
        cL = self.sound_speed(W_L)
        cR = self.sound_speed(W_R)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        pL = self.k * rhoL**self.gamma
        pR = self.k * rhoR**self.gamma
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        S_star = (
            0.5 * (uL + uR)
            if abs(denom) < self.min_value
            else (pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / denom
        )

        # Compute intermediate states
        rho_star_L = max(
            rhoL * (S_L - uL) / (S_L - S_star + self.min_value), self.min_value
        )
        rho_star_R = max(
            rhoR * (S_R - uR) / (S_R - S_star + self.min_value), self.min_value
        )
        U_L_star = np.array([rho_star_L, rho_star_L * S_star])
        U_R_star = np.array([rho_star_R, rho_star_R * S_star])

        # Compute flux
        FL = self.compute_flux(U_L)
        FR = self.compute_flux(U_R)
        if S_L >= 0:
            F = FL
        elif S_L <= 0 <= S_star:
            F = FL + S_L * (U_L_star - U_L)
        elif S_star <= 0 <= S_R:
            F = FR + S_R * (U_R_star - U_R)
        else:
            F = FR

        return F
