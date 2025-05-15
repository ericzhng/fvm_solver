import numpy as np

class GodunovSolver:
    def __init__(self, equation_system, flux_method='HLLC', limiter='minmod', reconstruction_method='muscl', cfl=0.5, bc_type='transmissive'):
        self.equation_system = equation_system
        self.flux_method = flux_method
        self.limiter = limiter
        self.reconstruction_method = reconstruction_method
        self.cfl = cfl
        self.bc_type = bc_type

    def minmod(self, a, b):
        if a * b > 0:
            return np.sign(a) * min(abs(a), abs(b))
        return 0

    def superbee(self, a, b):
        if a * b > 0:
            return np.sign(a) * max(min(2 * abs(a), abs(b)), min(abs(a), 2 * abs(b)))
        return 0

    def van_leer(self, a, b):
        if a * b > 0:
            return (a * b + abs(a * b)) / (a + b + 1e-10)
        return 0

    def get_limiter(self, a, b):
        if self.limiter == 'minmod':
            return self.minmod(a, b)
        elif self.limiter == 'superbee':
            return self.superbee(a, b)
        elif self.limiter == 'van_leer':
            return self.van_leer(a, b)
        else:
            raise ValueError("Unsupported limiter")

    def compute_cfl_dt(self, U, dx):
        n = len(U[0])
        max_speed = 0
        for i in range(n):
            W = self.equation_system.to_primitive(U[:, i])
            c = self.equation_system.sound_speed(W)
            max_speed = max(max_speed, abs(W[1]) + c)
        dt = self.cfl * dx / max_speed
        return dt

    def lax_friedrichs_flux(self, UL, UR):
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        if UL[0] <= 0 or UR[0] <= 0:
            u_avg = 0.5 * (W_L[1] + W_R[1])
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        FL = self.equation_system.compute_flux(UL, W_L)
        FR = self.equation_system.compute_flux(UR, W_R)
        lambda_max = max(abs(W_L[1]) + self.equation_system.sound_speed(W_L),
                         abs(W_R[1]) + self.equation_system.sound_speed(W_R))
        return 0.5 * (FL + FR) - 0.5 * lambda_max * (UR - UL)

    def rusanov_flux(self, UL, UR):
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        if UL[0] <= 0 or UR[0] <= 0:
            u_avg = 0.5 * (W_L[1] + W_R[1])
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        FL = self.equation_system.compute_flux(UL, W_L)
        FR = self.equation_system.compute_flux(UR, W_R)
        lambda_local = max(abs(W_L[1]) + self.equation_system.sound_speed(W_L),
                          abs(W_R[1]) + self.equation_system.sound_speed(W_R))
        return 0.5 * (FL + FR) - 0.5 * lambda_local * (UR - UL)

    def force_flux(self, UL, UR):
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        if UL[0] <= 0 or UR[0] <= 0:
            u_avg = 0.5 * (W_L[1] + W_R[1])
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        FL = self.equation_system.compute_flux(UL, W_L)
        FR = self.equation_system.compute_flux(UR, W_R)
        lambda_max = max(abs(W_L[1]) + self.equation_system.sound_speed(W_L),
                         abs(W_R[1]) + self.equation_system.sound_speed(W_R))
        F_LF = 0.5 * (FL + FR) - 0.5 * lambda_max * (UR - UL)
        U_mid = 0.5 * (UL + UR) - 0.5 * (FR - FL) / lambda_max
        W_mid = self.equation_system.to_primitive(U_mid)
        F_Richtmyer = self.equation_system.compute_flux(U_mid, W_mid)
        return 0.5 * (F_LF + F_Richtmyer)

    def hll_flux(self, UL, UR):
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        if UL[0] <= 0 or UR[0] <= 0:
            u_avg = 0.5 * (W_L[1] + W_R[1])
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
        else:
            return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)

    def hllc_flux(self, UL, UR):
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        if UL[0] <= 0 or UR[0] <= 0:
            u_avg = 0.5 * (W_L[1] + W_R[1])
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        cL = self.equation_system.sound_speed(W_L)
        cR = self.equation_system.sound_speed(W_R)
        SL = min(W_L[1] - cL, W_R[1] - cR)
        SR = max(W_L[1] + cL, W_R[1] + cR)
        if isinstance(self.equation_system, ShallowWaterSystem):
            hL, uL = W_L
            hR, uR = W_R
            S_star = (hR * uR * (SR - uR) - hL * uL * (SL - uL) + 0.5 * self.equation_system.g * (hR**2 - hL**2)) / \
                     (hR * (SR - uR) - hL * (SL - uL))
            hL_star = max(hL * (SL - uL) / (SL - S_star), min_var)
            hR_star = max(hR * (SR - uR) / (SR - S_star), min_var)
            UL_star = np.array([hL_star, hL_star * S_star])
            UR_star = np.array([hR_star, hR_star * S_star])
        else:  # EulerEquationSystem
            rhoL, uL, pL = W_L
            rhoR, uR, pR = W_R
            S_star = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / \
                     (rhoL * (SL - uL) - rhoR * (SR - uR))
            rhoL_star = max(rhoL * (SL - uL) / (SL - S_star), min_var)
            rhoR_star = max(rhoR * (SR - uR) / (SR - S_star), min_var)
            EL = UL[2] / rhoL + (S_star - uL) * (S_star + pL / (rhoL * (SL - uL)))
            ER = UR[2] / rhoR + (S_star - uR) * (S_star + pR / (rhoR * (SR - uR)))
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
        else:
            return FR

    def roe_flux(self, UL, UR):
        W_L = self.equation_system.to_primitive(UL)
        W_R = self.equation_system.to_primitive(UR)
        min_var = getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10))
        if UL[0] <= 0 or UR[0] <= 0:
            u_avg = 0.5 * (W_L[1] + W_R[1])
            return self.equation_system.compute_flux(UL if u_avg >= 0 else UR, W_L if u_avg >= 0 else W_R)
        if isinstance(self.equation_system, ShallowWaterSystem):
            hL, uL = W_L
            hR, uR = W_R
            h_roe = np.sqrt(hL * hR)
            u_roe = (uL * np.sqrt(hL) + uR * np.sqrt(hR)) / (np.sqrt(hL) + np.sqrt(hR))
            c_roe = np.sqrt(self.equation_system.g * h_roe)
            lambda1 = u_roe - c_roe
            lambda2 = u_roe + c_roe
            delta = 0.1 * c_roe
            lambda1 = lambda1 if abs(lambda1) > delta else 0.5 * (lambda1 + np.sqrt(lambda1**2 + delta**2))
            lambda2 = lambda2 if abs(lambda2) > delta else 0.5 * (lambda2 + np.sqrt(lambda2**2 + delta**2))
            R1 = np.array([1, u_roe - c_roe])
            R2 = np.array([1, u_roe + c_roe])
            delta_U = UR - UL
            alpha2 = (delta_U[0] * (u_roe - c_roe) - delta_U[1]) / (-2 * c_roe)
            alpha1 = delta_U[0] - alpha2
            FL = self.equation_system.compute_flux(UL, W_L)
            FR = self.equation_system.compute_flux(UR, W_R)
            return 0.5 * (FL + FR) - 0.5 * (abs(lambda1) * alpha1 * R1 + abs(lambda2) * alpha2 * R2)
        else:  # EulerEquationSystem
            rhoL, uL, pL = W_L
            rhoR, uR, pR = W_R
            hL = (UL[2] + pL) / rhoL
            hR = (UR[2] + pR) / rhoR
            rho_roe = np.sqrt(rhoL * rhoR)
            u_roe = (uL * np.sqrt(rhoL) + uR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR))
            h_roe = (hL * np.sqrt(rhoL) + hR * np.sqrt(rhoR)) / (np.sqrt(rhoL) + np.sqrt(rhoR))
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
            alpha2 = ((self.equation_system.gamma - 1) / c_roe**2) * \
                     (delta_rho * (0.5 * u_roe**2 - h_roe) + delta_rho_u * u_roe + delta_rho_E)
            alpha1 = ((delta_rho - alpha2) * (u_roe + c_roe) - delta_rho_u) / (2 * c_roe)
            alpha3 = (delta_rho_u - (delta_rho - alpha2) * (u_roe - c_roe)) / (2 * c_roe)
            FL = self.equation_system.compute_flux(UL, W_L)
            FR = self.equation_system.compute_flux(UR, W_R)
            return 0.5 * (FL + FR) - 0.5 * (abs(lambda1) * alpha1 * R1 + abs(lambda2) * alpha2 * R2 + abs(lambda3) * alpha3 * R3)

    def piecewise_constant_reconstruction(self, U_ext, dx):
        n = len(U_ext[0])
        n_vars = len(U_ext)
        UL = U_ext[:, :-1].copy()
        UR = U_ext[:, 1:].copy()
        for i in range(n-1):
            UL[0, i] = max(UL[0, i], getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10)))
            UR[0, i] = max(UR[0, i], getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10)))
        return UL, UR

    def muscl_reconstruction(self, U_ext, dx):
        n = len(U_ext[0])
        n_vars = len(U_ext)
        UL = np.zeros_like(U_ext[:, :-1])
        UR = np.zeros_like(U_ext[:, :-1])
        slopes = np.zeros_like(U_ext)
        for i in range(1, n-1):
            for var in range(n_vars):
                left_slope = (U_ext[var, i] - U_ext[var, i-1]) / dx
                right_slope = (U_ext[var, i+1] - U_ext[var, i]) / dx
                slopes[var, i] = self.get_limiter(left_slope, right_slope)
        for i in range(n-1):
            for var in range(n_vars):
                UL[var, i] = U_ext[var, i] + 0.5 * dx * slopes[var, i]
                UR[var, i] = U_ext[var, i+1] - 0.5 * dx * slopes[var, i+1]
            UL[0, i] = max(UL[0, i], getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10)))
            UR[0, i] = max(UR[0, i], getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10)))
        return UL, UR

    def ppm_reconstruction(self, U_ext, dx):
        n = len(U_ext[0])
        n_vars = len(U_ext)
        UL = np.zeros_like(U_ext[:, :-1])
        UR = np.zeros_like(U_ext[:, :-1])
        for var in range(n_vars):
            u = U_ext[var, :]
            u_L = np.zeros(n)
            u_R = np.zeros(n)
            # Compute interface values
            for i in range(2, n-2):
                delta_m = self.minmod(u[i] - u[i-1], u[i+1] - u[i])
                u_L[i] = u[i] - 0.5 * delta_m
                u_R[i] = u[i] + 0.5 * delta_m
                # Monotonicity constraints
                if (u_R[i] - u_L[i]) * (u[i] - 0.5 * (u_L[i] + u_R[i])) > (u_R[i] - u_L[i])**2 / 6:
                    u_L[i] = 3 * u[i] - 2 * u_R[i]
                elif (u_R[i] - u_L[i]) * (u[i] - 0.5 * (u_L[i] + u_R[i])) < -(u_R[i] - u_L[i])**2 / 6:
                    u_R[i] = 3 * u[i] - 2 * u_L[i]
            # Interface states
            UL[var, 1:-1] = u_R[2:-2]
            UR[var, 1:-1] = u_L[3:-1]
            UL[var, 0] = u[2]
            UR[var, 0] = u[2]
            UL[var, -1] = u[-3]
            UR[var, -1] = u[-3]
        for i in range(n-1):
            UL[0, i] = max(UL[0, i], getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10)))
            UR[0, i] = max(UR[0, i], getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10)))
        return UL, UR

    def weno5_reconstruction(self, U_ext, dx):
        n = len(U_ext[0])
        n_vars = len(U_ext)
        UL = np.zeros_like(U_ext[:, :-1])
        UR = np.zeros_like(U_ext[:, :-1])
        eps = 1e-6
        for var in range(n_vars):
            u = U_ext[var, :]
            for i in range(2, n-2):
                # Left state (UR[i-1/2])
                v0 = u[i-2:i+1]  # Stencil: u[i-2], u[i-1], u[i]
                v1 = u[i-1:i+2]  # Stencil: u[i-1], u[i], u[i+1]
                v2 = u[i:i+3]    # Stencil: u[i], u[i+1], u[i+2]
                # Polynomials
                p0 = (2 * v0[0] - 7 * v0[1] + 11 * v0[2]) / 6
                p1 = (-v1[0] + 5 * v1[1] + 2 * v1[2]) / 6
                p2 = (2 * v2[0] + 5 * v2[1] - v2[2]) / 6
                # Smoothness indicators
                beta0 = 13/12 * (v0[0] - 2 * v0[1] + v0[2])**2 + 1/4 * (v0[0] - 4 * v0[1] + 3 * v0[2])**2
                beta1 = 13/12 * (v1[0] - 2 * v1[1] + v1[2])**2 + 1/4 * (v1[0] - v1[2])**2
                beta2 = 13/12 * (v2[0] - 2 * v2[1] + v2[2])**2 + 1/4 * (3 * v2[0] - 4 * v2[1] + v2[2])**2
                # Weights
                alpha0 = 0.1 / (beta0 + eps)**2
                alpha1 = 0.6 / (beta1 + eps)**2
                alpha2 = 0.3 / (beta2 + eps)**2
                w_sum = alpha0 + alpha1 + alpha2
                w0 = alpha0 / w_sum
                w1 = alpha1 / w_sum
                w2 = alpha2 / w_sum
                UL[var, i-1] = w0 * p0 + w1 * p1 + w2 * p2
                # Right state (UL[i+1/2])
                p0 = (-v2[2] + 5 * v2[1] + 2 * v2[0]) / 6
                p1 = (2 * v1[2] + 5 * v1[1] - v1[0]) / 6
                p2 = (11 * v0[2] - 7 * v0[1] + 2 * v0[0]) / 6
                beta0 = 13/12 * (v2[2] - 2 * v2[1] + v2[0])**2 + 1/4 * (v2[2] - 4 * v2[1] + 3 * v2[0])**2
                beta1 = 13/12 * (v1[2] - 2 * v1[1] + v1[0])**2 + 1/4 * (v1[2] - v1[0])**2
                beta2 = 13/12 * (v0[2] - 2 * v0[1] + v0[0])**2 + 1/4 * (3 * v0[2] - 4 * v0[1] + v0[0])**2
                alpha0 = 0.1 / (beta0 + eps)**2
                alpha1 = 0.6 / (beta1 + eps)**2
                alpha2 = 0.3 / (beta2 + eps)**2
                w_sum = alpha0 + alpha1 + alpha2
                w0 = alpha0 / w_sum
                w1 = alpha1 / w_sum
                w2 = alpha2 / w_sum
                UR[var, i-1] = w0 * p0 + w1 * p1 + w2 * p2
            # Boundary handling
            UL[var, 0] = u[2]
            UR[var, 0] = u[3]
            UL[var, -1] = u[-3]
            UR[var, -1] = u[-2]
        for i in range(n-1):
            UL[0, i] = max(UL[0, i], getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10)))
            UR[0, i] = max(UR[0, i], getattr(self.equation_system, 'h_min', getattr(self.equation_system, 'rho_min', 1e-10)))
        return UL, UR

    def apply_boundary_conditions(self, U):
        n = len(U[0])
        n_vars = len(U)
        U_ext = np.zeros((n_vars, n + 4))
        U_ext[:, 2:-2] = U
        if self.bc_type == 'transmissive':
            U_ext[:, 0:2] = U[:, 0:1]
            U_ext[:, -2:] = U[:, -1:]
        elif self.bc_type == 'reflective':
            U_ext[:, 0:2] = U[:, 0:1]
            U_ext[1, 0:2] = -U_ext[1, 0:2]
            U_ext[:, -2:] = U[:, -1:]
            U_ext[1, -2:] = -U_ext[1, -2:]
        elif self.bc_type == 'periodic':
            U_ext[:, 0:2] = U[:, -2:]
            U_ext[:, -2:] = U[:, 0:2]
        else:
            raise ValueError("Unsupported boundary condition")
        return U_ext

    def solve(self, U, dx, dt):
        n = len(U[0])
        n_vars = len(U)
        flux = np.zeros((n_vars, n + 1))
        U_new = np.zeros_like(U)
        U_ext = self.apply_boundary_conditions(U)
        if self.reconstruction_method == 'piecewise_constant':
            UL, UR = self.piecewise_constant_reconstruction(U_ext, dx)
        elif self.reconstruction_method == 'muscl':
            UL, UR = self.muscl_reconstruction(U_ext, dx)
        elif self.reconstruction_method == 'ppm':
            UL, UR = self.ppm_reconstruction(U_ext, dx)
        elif self.reconstruction_method == 'weno5':
            UL, UR = self.weno5_reconstruction(U_ext, dx)
        else:
            raise ValueError("Unsupported reconstruction method")
        for i in range(n + 1):
            if self.flux_method == 'Lax-Friedrichs':
                flux[:, i] = self.lax_friedrichs_flux(UL[:, i + 1], UR[:, i + 1])
            elif self.flux_method == 'Rusanov':
                flux[:, i] = self.rusanov_flux(UL[:, i + 1], UR[:, i + 1])
            elif self.flux_method == 'FORCE':
                flux[:, i] = self.force_flux(UL[:, i + 1], UR[:, i + 1])
            elif self.flux_method == 'HLL':
                flux[:, i] = self.hll_flux(UL[:, i + 1], UR[:, i + 1])
            elif self.flux_method == 'HLLC':
                flux[:, i] = self.hllc_flux(UL[:, i + 1], UR[:, i + 1])
            elif self.flux_method == 'Roe':
                flux[:, i] = self.roe_flux(UL[:, i + 1], UR[:, i + 1])
            else:
                raise ValueError("Unsupported flux method")
        for i in range(n):
            U_new[:, i] = U[:, i] - dt / dx * (flux[:, i + 1] - flux[:, i])
        return U_new

    def plot_variable_evolution(self, snapshots, times, x):
        import matplotlib.pyplot as plt
        var_name = self.equation_system.get_variable_names()[0]
        plt.figure(figsize=(10, 6))
        for t, var in zip(times, snapshots):
            plt.plot(x, var, label=f't = {t:.3f}', alpha=0.7)
        plt.title(f'{var_name} Evolution ({self.equation_system.__class__.__name__}, {self.reconstruction_method}, {self.flux_method})')
        plt.xlabel('x')
        plt.ylabel(var_name)
        plt.grid(True)
        plt.legend()
        plt.savefig(f'variable_evolution_{self.equation_system.__class__.__name__}_{self.reconstruction_method}_{self.flux_method}.png')
        plt.close()