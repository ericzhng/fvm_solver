import numpy as np
from .equation import ShallowWaterSystem, EulerEquationSystem

class Flux:
    """Computes numerical fluxes for hyperbolic conservation laws."""
    
    def __init__(self, equation_system):
        """Initialize flux computation with equation system.
        
        Args:
            equation_system: Equation system instance
        """
        self.equation_system = equation_system
        self.min_var = getattr(equation_system, 'h_min', 
                               getattr(equation_system, 'rho_min', 
                                       getattr(equation_system, 'p_min', 1e-10)))

    def lax_friedrichs(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Lax-Friedrichs flux: Simple, diffusive.
        
        Args:
            UL: Left state
            UR: Right state
            
        Returns:
            Numerical flux
        """
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)

        u_L = self.equation_system.velocity(W_L)
        u_R = self.equation_system.velocity(W_R)

        if UL[0] <= self.min_var or UR[0] <= self.min_var:
            u_avg = 0.5 * (u_L + u_R)
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        
        FL = self.equation_system.compute_flux(UL, W_L)
        FR = self.equation_system.compute_flux(UR, W_R)

        lambda_max = max(abs(u_L) + self.equation_system.sound_speed(W_L),
                         abs(u_R) + self.equation_system.sound_speed(W_R))

        return 0.5 * (FL + FR) - 0.5 * lambda_max * (UR - UL)

    def rusanov(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Rusanov flux: Local Lax-Friedrichs, less diffusive.
        
        Args:
            UL: Left state
            UR: Right state
            
        Returns:
            Numerical flux
        """
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        
        u_L = self.equation_system.velocity(W_L)
        u_R = self.equation_system.velocity(W_R)

        if UL[0] <= self.min_var or UR[0] <= self.min_var:
            u_avg = 0.5 * (u_L + u_R)
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        FL = self.equation_system.compute_flux(UL, W_L)
        FR = self.equation_system.compute_flux(UR, W_R)
        
        lambda_local = max(abs(u_L) + self.equation_system.sound_speed(W_L),
                         abs(u_R) + self.equation_system.sound_speed(W_R))

        return 0.5 * (FL + FR) - 0.5 * lambda_local * (UR - UL)

    def force(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """FORCE flux: Combines Lax-Friedrichs and Richtmyer.
        
        Args:
            UL: Left state
            UR: Right state
            
        Returns:
            Numerical flux
        """
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)

        u_L = self.equation_system.velocity(W_L)
        u_R = self.equation_system.velocity(W_R)

        if UL[0] <= self.min_var or UR[0] <= self.min_var:
            u_avg = 0.5 * (u_L + u_R)
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        FL = self.equation_system.compute_flux(UL, W_L)
        FR = self.equation_system.compute_flux(UR, W_R)
        lambda_max = max(abs(W_L[1]) + self.equation_system.sound_speed(W_L),
                         abs(W_R[1]) + self.equation_system.sound_speed(W_R))
        F_LF = 0.5 * (FL + FR) - 0.5 * lambda_max * (UR - UL)
        U_mid = 0.5 * (UL + UR) - 0.5 * (FR - FL) / (lambda_max + 1e-10)
        W_mid = self.equation_system.to_primitive(U_mid)
        F_Richtmyer = self.equation_system.compute_flux(U_mid, W_mid)
        return 0.5 * (F_LF + F_Richtmyer)

    def hll(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """HLL flux: Uses two-wave approximation.
        
        Args:
            UL: Left state
            UR: Right state
            
        Returns:
            Numerical flux
        """
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        
        u_L = self.equation_system.velocity(W_L)
        u_R = self.equation_system.velocity(W_R)

        if UL[0] <= self.min_var or UR[0] <= self.min_var:
            u_avg = 0.5 * (u_L + u_R)
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        
        cL = self.equation_system.sound_speed(W_L)
        cR = self.equation_system.sound_speed(W_R)
        SL = min(W_L[1] - cL, W_R[1] - cR)
        SR = max(W_L[1] + cL, W_R[1] + cR)
        FL = self.equation_system.compute_flux(UL, W_L)
        FR = self.equation_system.compute_flux(UR, W_R)
        if SL >= 0:
            return FL
        elif SR <= 0:
            return FR
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + 1e-10)

    def hllc(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """HLLC flux: Includes contact wave for better resolution.
        
        Args:
            UL: Left state
            UR: Right state
            
        Returns:
            Numerical flux
        """
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        
        u_L = self.equation_system.velocity(W_L)
        u_R = self.equation_system.velocity(W_R)

        if UL[0] <= self.min_var or UR[0] <= self.min_var:
            u_avg = 0.5 * (u_L + u_R)
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        cL = self.equation_system.sound_speed(W_L)
        cR = self.equation_system.sound_speed(W_R)
        SL = min(W_L[1] - cL, W_R[1] - cR)
        SR = max(W_L[1] + cL, W_R[1] + cR)
        if isinstance(self.equation_system, ShallowWaterSystem):
            hL, uL = W_L
            hR, uR = W_R
            denom = hR * (SR - uR) - hL * (SL - uL)
            S_star = 0.5 * (uL + uR) if abs(denom) < 1e-10 else \
                     (hR * uR * (SR - uR) - hL * uL * (SL - uL) + 0.5 * self.equation_system.g * (hR**2 - hL**2)) / denom
            hL_star = max(hL * (SL - uL) / (SL - S_star + 1e-10), self.min_var)
            hR_star = max(hR * (SR - uR) / (SR - S_star + 1e-10), self.min_var)
            UL_star = np.array([hL_star, hL_star * S_star])
            UR_star = np.array([hR_star, hR_star * S_star])
        elif isinstance(self.equation_system, EulerEquationSystem):
            rhoL, uL, pL = W_L
            rhoR, uR, pR = W_R
            denom = rhoL * (SL - uL) - rhoR * (SR - uR)
            S_star = 0.5 * (uL + uR) if abs(denom) < 1e-10 else \
                     (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denom
            rhoL_star = max(rhoL * (SL - uL) / (SL - S_star + 1e-10), self.min_var)
            rhoR_star = max(rhoR * (SR - uR) / (SR - S_star + 1e-10), self.min_var)
            EL = UL[2] / (rhoL + 1e-10) + (S_star - uL) * (S_star + pL / (rhoL * (SL - uL) + 1e-10))
            ER = UR[2] / (rhoR + 1e-10) + (S_star - uR) * (S_star + pR / (rhoR * (SR - uR) + 1e-10))
            UL_star = np.array([rhoL_star, rhoL_star * S_star, rhoL_star * EL])
            UR_star = np.array([rhoR_star, rhoR_star * S_star, rhoR_star * ER])
        FL = self.equation_system.compute_flux(UL, W_L)
        FR = self.equation_system.compute_flux(UR, W_R)
        if SL >= 0:
            return FL
        elif SL <= 0 <= S_star:
            return FL + SL * (UL_star - UL)
        elif S_star <= 0 <= SR:
            return FR + SR * (UR_star - UR)
        return FR

    def roe(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Roe flux: Linearized Riemann solver with entropy fix.
        
        Args:
            UL: Left state
            UR: Right state
            
        Returns:
            Numerical flux
        """
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        
        u_L = self.equation_system.velocity(W_L)
        u_R = self.equation_system.velocity(W_R)

        if UL[0] <= self.min_var or UR[0] <= self.min_var:
            u_avg = 0.5 * (u_L + u_R)
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        if isinstance(self.equation_system, ShallowWaterSystem):
            hL, uL = W_L
            hR, uR = W_R
            h_roe = np.sqrt(hL * hR)
            u_roe = (uL * np.sqrt(hL) + uR * np.sqrt(hR)) / (np.sqrt(hL) + np.sqrt(hR) + 1e-10)
            c_roe = np.sqrt(self.equation_system.g * h_roe)
            lambda1 = u_roe - c_roe
            lambda2 = u_roe + c_roe
            delta = 0.1 * c_roe
            lambda1 = lambda1 if abs(lambda1) > delta else 0.5 * (lambda1 + np.sqrt(lambda1**2 + delta**2))
            lambda2 = lambda2 if abs(lambda2) > delta else 0.5 * (lambda2 + np.sqrt(lambda2**2 + delta**2))
            R1 = np.array([1, u_roe - c_roe])
            R2 = np.array([1, u_roe + c_roe])
            delta_U = UR - UL
            alpha2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe + 1e-10)
            alpha1 = delta_U[0] - alpha2
            FL = self.equation_system.compute_flux(UL, W_L)
            FR = self.equation_system.compute_flux(UR, W_R)
            return 0.5 * (FL + FR) - 0.5 * (abs(lambda1) * alpha1 * R1 + abs(lambda2) * alpha2 * R2)
        elif isinstance(self.equation_system, EulerEquationSystem):
            rhoL, uL, pL = W_L
            rhoR, uR, pR = W_R
            hL = (UL[2] + pL) / (rhoL + 1e-10)
            hR = (UR[2] + pR) / (rhoR + 1e-10)
            rho_roe = np.sqrt(rhoL * rhoR)
            u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + 1e-10)
            h_roe = (hL * np.sqrt(rhoL) + hR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR) + 1e-10)
            c_roe = np.sqrt((self.equation_system.gamma - 1) * (h_roe - 0.5 * u_roe**2))
            lambda1 = u_roe - c_roe
            lambda2 = u_roe
            lambda3 = u_roe + c_roe
            delta = 0.1 * c_roe
            lambda1 = lambda1 if abs(lambda1) > delta else 0.5 * (lambda1 + np.sqrt(lambda1**2 + delta**2))
            lambda2 = lambda2 if abs(lambda2) > delta else 0.5 * (lambda2 + np.sqrt(lambda2**2 + delta**2))
            lambda3 = lambda3 if abs(lambda3) > delta else 0.5 * (lambda3 + np.sqrt(lambda3**2 + delta**2))
            R1 = np.array([1, u_roe - c_roe, h_roe - u_roe * c_roe])
            R2 = np.array([1, u_roe, 0.5 * u_roe**2])
            R3 = np.array([1, u_roe + c_roe, h_roe + u_roe * c_roe])
            delta_U = UR - UL
            delta_rho = delta_U[0]
            delta_rho_u = delta_U[1]
            delta_rho_E = delta_U[2]
            alpha2 = ((self.equation_system.gamma - 1) / (c_roe**2 + 1e-10)) * \
                     (delta_rho * (0.5 * u_roe**2 - h_roe) + delta_rho_u * u_roe + delta_rho_E)
            alpha1 = ((delta_rho - alpha2) * (u_roe + c_roe) - delta_rho_u) / (2 * c_roe + 1e-10)
            alpha3 = (delta_rho_u - (delta_rho - alpha2) * (u_roe - c_roe)) / (2 * c_roe + 1e-10)
            FL = self.equation_system.compute_flux(UL, W_L)
            FR = self.equation_system.compute_flux(UR, W_R)
            return 0.5 * (FL + FR) - 0.5 * (abs(lambda1) * alpha1 * R1 + abs(lambda2) * alpha2 * R2 + abs(lambda3) * alpha3 * R3)

    def get_flux(self, name: str):
        """Returns the flux function by name.
        
        Args:
            name: Flux method name (e.g., 'HLLC', 'Roe')
            
        Returns:
            Flux function
            
        Raises:
            ValueError: If flux name is unsupported
        """
        fluxes = {
            'Lax-Friedrichs': self.lax_friedrichs,
            'Rusanov': self.rusanov,
            'FORCE': self.force,
            'HLL': self.hll,
            'HLLC': self.hllc,
            'Roe': self.roe
        }
        if name not in fluxes:
            raise ValueError(f"Unsupported flux: {name}")
        return fluxes[name]
