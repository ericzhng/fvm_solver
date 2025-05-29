import numpy as np

class GodunovSolver:
    def __init__(self, gamma=1.4, flux_method='HLLC', limiter=None):
        self.gamma = gamma
        self.flux_method = flux_method
        self.limiter = limiter

    def minmod(self, a, b):
        """Minmod limiter."""
        if a * b > 0:
            return np.sign(a) * min(abs(a), abs(b))
        return 0

    def superbee(self, a, b):
        """Superbee limiter."""
        if a * b > 0:
            return np.sign(a) * max(min(2 * abs(a), abs(b)), min(abs(a), 2 * abs(b)))
        return 0

    def get_limiter(self, a, b):
        """Select limiter."""
        if self.limiter == 'minmod':
            return self.minmod(a, b)
        elif self.limiter == 'superbee':
            return self.superbee(a, b)
        return 0  # No limiter for Godunov's method

    def primitive_to_conservative(self, rho, u, p):
        """Convert primitive to conservative variables."""
        mom = rho * u
        energy = p / (self.gamma - 1) + 0.5 * rho * u**2
        return np.array([rho, mom, energy])

    def conservative_to_primitive(self, U):
        """Convert conservative to primitive variables."""
        rho = U[0]
        u = U[1] / (rho + 1E-10)
        p = (self.gamma - 1) * (U[2] - 0.5 * rho * u**2)
        return np.array([rho, u, p])

    def sound_speed(self, rho, p):
        """Calculate sound speed."""
        return np.sqrt(self.gamma * p / (rho + 1E-10))

    def hllc_flux(self, UL, UR):
        """HLLC Riemann solver flux."""
        # Primitive variables
        rhoL, uL, pL = self.conservative_to_primitive(UL)
        rhoR, uR, pR = self.conservative_to_primitive(UR)

        # Sound speeds
        cL = self.sound_speed(rhoL, pL)
        cR = self.sound_speed(rhoR, pR)

        # Wave speeds
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)

        # Star region
        S_star = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / \
                 (rhoL * (SL - uL) - rhoR * (SR - uR) + 1E-10)

        # Star states
        rhoL_star = rhoL * (SL - uL) / (SL - S_star)
        rhoR_star = rhoR * (SR - uR) / (SR - S_star)

        EL = UL[2] / rhoL + (S_star - uL) * (S_star + pL / (rhoL * (SL - uL)))
        ER = UR[2] / rhoR + (S_star - uR) * (S_star + pR / (rhoR * (SR - uR)))

        UL_star = np.array([rhoL_star, rhoL_star * S_star, rhoL_star * EL])
        UR_star = np.array([rhoR_star, rhoR_star * S_star, rhoR_star * ER])

        # Fluxes
        FL = self.compute_flux(UL, [rhoL, uL, pL])
        FR = self.compute_flux(UR, [rhoR, uR, pR])

        if SL >= 0:
            return FL
        elif SL <= 0 <= S_star:
            return FL + SL * (UL_star - UL)
        elif S_star <= 0 <= SR:
            return FR + SR * (UR_star - UR)
        else:
            return FR

    def compute_flux(self, U, W):
        """Compute physical flux."""
        rho, u, p = W
        mom = U[1]
        energy = U[2]
        return np.array([mom, rho * u**2 + p, u * (energy + p)])

    def solve(self, U, dx, dt):
        """Godunov's method with Riemann solver."""
        n = len(U[0])
        flux = np.zeros_like(U)
        U_new = np.zeros_like(U)

        # Godunov: piecewise constant (no reconstruction unless limiter specified)
        UL = U[:, :-1]
        UR = U[:, 1:]

        if self.limiter:
            slopes = np.zeros_like(U)
            for i in range(1, n-1):
                for var in range(3):
                    slopes[var, i] = self.get_limiter(
                        (U[var, i] - U[var, i-1]) / dx,
                        (U[var, i+1] - U[var, i]) / dx
                    )
            for i in range(n-1):
                for var in range(3):
                    UL[var, i] = U[var, i] + 0.5 * dx * slopes[var, i]
                    UR[var, i] = U[var, i+1] - 0.5 * dx * slopes[var, i+1]

        # Compute fluxes
        for i in range(n-1):
            if self.flux_method == 'HLLC':
                flux[:, i] = self.hllc_flux(UL[:, i], UR[:, i])
            else:
                raise ValueError("Unsupported flux method")

        # Update solution
        for i in range(1, n-1):
            U_new[:, i] = U[:, i] - dt / dx * (flux[:, i] - flux[:, i-1])

        return U_new

# Example usage
if __name__ == "__main__":
    # Setup
    n_cells = 100
    dx = 1.0 / n_cells
    dt = 0.001
    gamma = 1.4

    # Initial conditions (Sod shock tube)
    U = np.zeros((3, n_cells))
    x = np.linspace(0, 1, n_cells)
    for i in range(n_cells):
        if x[i] < 0.5:
            U[:, i] = np.array([1.0, 0.0, 1.0 / (gamma - 1)])
        else:
            U[:, i] = np.array([0.125, 0.0, 0.1 / (gamma - 1)])

    solver = GodunovSolver(gamma=gamma, flux_method='HLLC', limiter=None)

    # Time stepping
    t_final = 0.1
    t = 0
    while t < t_final:
        U = solver.solve(U, dx, dt)
        t += dt

    # Output results
    print("Final density:", U[0])