import numpy as np
from .base_equation import EquationSystem


class EulerEquation(EquationSystem):
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
        self.vel_idx = 1
        self.monitor_idx = 0
        self.var_names = ['density', 'velocity', 'pressure']
        self.num_vars = 3
        self.safety_guard_var_idx = [0, 2]  # density and pressure

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

    def sound_speed(self, W: np.ndarray) -> float:
        """Compute the sound speed for the gas.

        Args:
            W (np.ndarray): [density, velocity, pressure]

        Returns:
            float: Sound speed (sqrt(gamma * p / rho))
        """
        if W.shape != (self.num_vars,):
            raise ValueError(f"W must have shape ({self.num_vars},)")
        rho, _, p = W
        rho = np.maximum(rho, self.min_value)
        p = np.maximum(p, self.min_value)
        return np.sqrt(self.gamma * p / rho)

    def compute_flux(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute the physical flux.

        Args:
            U (np.ndarray): [density, momentum, total energy]
            W (np.ndarray): [density, velocity, pressure]

        Returns:
            np.ndarray: [rho*u, rho*u^2 + p, u*(E + p)]
        """
        if U.shape != (self.num_vars,) or W.shape != (self.num_vars,):
            raise ValueError(f"U and W must have shape ({self.num_vars},)")
        rho, u, p = W
        E = U[2]
        return np.array([rho * u, rho * u**2 + p, u * (E + p)])

    def hllc_numerical_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute HLLC numerical flux for Euler equations.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity, pressure]
            WR (np.ndarray): Right primitive state [density, velocity, pressure]
            UL (np.ndarray): Left conservative state [density, momentum, energy]
            UR (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            np.ndarray: HLLC numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # Extract and ensure positivity
        rhoL, uL, pL = WL
        rhoR, uR, pR = WR
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)
        pL = np.maximum(pL, self.min_value)
        pR = np.maximum(pR, self.min_value)

        # Wave speeds
        cL = np.sqrt(self.gamma * pL / rhoL)
        cR = np.sqrt(self.gamma * pR / rhoR)
        S_L = min(uL - cL, uR - cR)
        S_R = max(uL + cL, uR + cR)
        denom = rhoL * (S_L - uL) - rhoR * (S_R - uR)
        if abs(denom) < self.min_value:
            S_star = 0.5 * (uL + uR)
        else:
            S_star = (pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / denom

        # Intermediate states
        rhoL_star = max(rhoL * (S_L - uL) / (S_L - S_star + self.min_value), self.min_value)
        rhoR_star = max(rhoR * (S_R - uR) / (S_R - S_star + self.min_value), self.min_value)
        EL = UL[2] / (rhoL + self.min_value) + (S_star - uL) * (
            S_star + pL / (rhoL * (S_L - uL) + self.min_value))
        ER = UR[2] / (rhoR + self.min_value) + (S_star - uR) * (
            S_star + pR / (rhoR * (S_R - uR) + self.min_value))
        UL_star = np.array([rhoL_star, rhoL_star * S_star, rhoL_star * EL])
        UR_star = np.array([rhoR_star, rhoR_star * S_star, rhoR_star * ER])

        # Fluxes
        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)
        if S_L >= 0:
            F = FL
        elif S_L <= 0 <= S_star:
            F = FL + S_L * (UL_star - UL)
        elif S_star <= 0 <= S_R:
            F = FR + S_R * (UR_star - UR)
        else:
            F = FR

        return F

    def roe_numerical_flux(self, WL: np.ndarray, WR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Compute the Roe numerical flux for the shallow water equations, with entropy fix.

        Args:
            WL (np.ndarray): Left primitive state [density, velocity, pressure]
            WR (np.ndarray): Right primitive state [density, velocity, pressure]
            UL (np.ndarray): Left conservative state [density, momentum, energy]
            UR (np.ndarray): Right conservative state [density, momentum, energy]

        Returns:
            np.ndarray: Roe numerical flux
        """
        if any(arr.shape != (self.num_vars,) for arr in [WL, WR, UL, UR]):
            raise ValueError(f"All inputs must have shape ({self.num_vars},)")

        # Extract left and right states
        rhoL, uL, pL = WL
        rhoR, uR, pR = WR
        rhoL = np.maximum(rhoL, self.min_value)
        rhoR = np.maximum(rhoR, self.min_value)
        pL = np.maximum(pL, self.min_value)
        pR = np.maximum(pR, self.min_value)

        # Compute enthalpy
        EL = UL[2]
        ER = UR[2]
        hL = (EL + pL) / rhoL
        hR = (ER + pR) / rhoR

        # Roe averages
        sqrt_rhoL = np.sqrt(rhoL)
        sqrt_rhoR = np.sqrt(rhoR)
        denom = sqrt_rhoL + sqrt_rhoR + self.min_value
        u_roe = (uL * sqrt_rhoL + uR * sqrt_rhoR) / denom
        h_roe = (hL * sqrt_rhoL + hR * sqrt_rhoR) / denom
        c2_roe = (self.gamma - 1) * (h_roe - 0.5 * u_roe**2)
        c2_roe = np.maximum(c2_roe, self.min_value)
        c_roe = np.sqrt(c2_roe)

        # Eigenvalues
        lambda_1 = u_roe - c_roe
        lambda_2 = u_roe
        lambda_3 = u_roe + c_roe
        lambdas = np.array([lambda_1, lambda_2, lambda_3])

        # Improved entropy fix (Harten-Hyman)
        delta = 0.1 * c_roe
        abs_lambdas = np.abs(lambdas)
        for i in range(3):
            if abs_lambdas[i] < delta:
                abs_lambdas[i] = 0.5 * (lambdas[i] + np.sqrt(lambdas[i]**2 + delta**2))
                # abs_lambdas[i] = 0.5 * (lambdas[i]**2 / delta + delta)
                
        # Differences
        delta_U = UR - UL

        # Compute wave strengths (alpha) using conservative variable jumps
        delta_rho = delta_U[0]
        delta_rho_u = delta_U[1]
        delta_rho_E = delta_U[2]
        alpha_2 = ((self.gamma - 1) / (c_roe**2 + self.min_value)) * (
            delta_rho * (0.5 * u_roe**2 - h_roe) + delta_rho_u * u_roe + delta_rho_E
        )
        alpha_1 = ((delta_rho - alpha_2) * (u_roe + c_roe) - delta_rho_u) / (2 * c_roe + self.min_value)
        alpha_3 = (delta_rho_u - (delta_rho - alpha_2) * (u_roe - c_roe)) / (2 * c_roe + self.min_value)

        # Roe eigenvectors
        r1 = np.array([1, u_roe - c_roe, h_roe - u_roe * c_roe])
        r2 = np.array([1, u_roe, 0.5 * u_roe**2])
        r3 = np.array([1, u_roe + c_roe, h_roe + u_roe * c_roe])

        # |A| * delta_U
        dissipative_term = abs_lambdas[0] * alpha_1 * r1 + abs_lambdas[1] * alpha_2 * r2 + abs_lambdas[2] * alpha_3 * r3

        # Physical fluxes
        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)

        # Roe flux
        F = 0.5 * (FL + FR) - 0.5 * dissipative_term

        # Pressure jump detector
        pL, pR = WL[2], WR[2]
        pressure_jump = max(pL, pR) / (min(pL, pR) + self.min_value)
        if pressure_jump > 2.0:
            # Fallback to Rusanov flux for this interface
            FL_rusanov = FL
            FR_rusanov = FR
            lambda_local = max(
                abs(WL[self.vel_idx]) + self.sound_speed(WL),
                abs(WR[self.vel_idx]) + self.sound_speed(WR)
            )
            F = 0.5 * (FL_rusanov + FR_rusanov - lambda_local * (UR - UL))
        
        # Optionally, add a small artificial viscosity term
        shock_indicator = abs(pR - pL) / (pR + pL + self.min_value)
        if shock_indicator > 0.2:  # tune threshold as needed
            F += 0.2 * (UR - UL)  # add small viscosity
            
        return F
