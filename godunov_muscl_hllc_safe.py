import numpy as np
import matplotlib.pyplot as plt

class GodunovSolver:
    def __init__(self, gamma=1.4, flux_method='HLLC', limiter='minmod', cfl=0.5, rho_min=1e-10, p_min=1e-10, bc_type='transmissive'):
        self.gamma = gamma
        self.flux_method = flux_method
        self.limiter = limiter
        self.cfl = cfl
        self.rho_min = rho_min
        self.p_min = p_min
        self.bc_type = bc_type

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
        else:
            raise ValueError("Unsupported limiter")

    def primitive_to_conservative(self, rho, u, p):
        """Convert primitive to conservative variables with safeguards."""
        rho = max(rho, self.rho_min)
        p = max(p, self.p_min)
        mom = rho * u
        energy = p / (self.gamma - 1) + 0.5 * rho * u**2
        return np.array([rho, mom, energy])

    def conservative_to_primitive(self, U):
        """Convert conservative to primitive variables with safeguards."""
        rho = max(U[0], self.rho_min)
        u = U[1] / rho
        p = max((self.gamma - 1) * (U[2] - 0.5 * rho * u**2), self.p_min)
        return np.array([rho, u, p])

    def sound_speed(self, rho, p):
        """Calculate sound speed with safeguards."""
        rho = max(rho, self.rho_min)
        p = max(p, self.p_min)
        return np.sqrt(self.gamma * p / rho)

    def compute_cfl_dt(self, U, dx):
        """Compute time step based on CFL condition."""
        n = len(U[0])
        max_speed = 0
        for i in range(n):
            rho, u, p = self.conservative_to_primitive(U[:, i])
            c = self.sound_speed(rho, p)
            max_speed = max(max_speed, abs(u) + c)
        dt = self.cfl * dx / max_speed
        return dt

    def hllc_flux(self, UL, UR):
        """HLLC Riemann solver flux with safeguards."""
        if UL[0] <= 0 or UR[0] <= 0:
            rhoL, uL, pL = self.conservative_to_primitive(UL)
            rhoR, uR, pR = self.conservative_to_primitive(UR)
            u_avg = 0.5 * (uL + uR)
            return self.compute_flux(UL if u_avg >= 0 else UR, 
                                   [rhoL, uL, pL] if u_avg >= 0 else [rhoR, uR, pR])
        rhoL, uL, pL = self.conservative_to_primitive(UL)
        rhoR, uR, pR = self.conservative_to_primitive(UR)
        cL = self.sound_speed(rhoL, pL)
        cR = self.sound_speed(rhoR, pR)
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)
        S_star = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / \
                 (rhoL * (SL - uL) - rhoR * (SR - uR))
        rhoL_star = rhoL * (SL - uL) / (SL - S_star)
        rhoR_star = rhoR * (SR - uR) / (SR - S_star)
        rhoL_star = max(rhoL_star, self.rho_min)
        rhoR_star = max(rhoR_star, self.rho_min)
        EL = UL[2] / rhoL + (S_star - uL) * (S_star + pL / (rhoL * (SL - uL)))
        ER = UR[2] / rhoR + (S_star - uR) * (S_star + pR / (rhoR * (SR - uR)))
        UL_star = np.array([rhoL_star, rhoL_star * S_star, rhoL_star * EL])
        UR_star = np.array([rhoR_star, rhoR_star * S_star, rhoR_star * ER])
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
        """Compute physical flux with safeguards."""
        rho, u, p = W
        rho = max(rho, self.rho_min)
        p = max(p, self.p_min)
        mom = U[1]
        energy = U[2]
        return np.array([mom, rho * u**2 + p, u * (energy + p)])

    def apply_boundary_conditions(self, U):
        """Apply boundary conditions using ghost cells."""
        n = len(U[0])
        U_ext = np.zeros((3, n + 4))
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

    def muscl_reconstruction(self, U_ext, dx):
        """MUSCL reconstruction with slope limiters and density clipping."""
        n = len(U_ext[0])
        UL = np.zeros_like(U_ext[:, :-1])
        UR = np.zeros_like(U_ext[:, :-1])
        slopes = np.zeros_like(U_ext)
        for i in range(1, n-1):
            for var in range(3):
                left_slope = (U_ext[var, i] - U_ext[var, i-1]) / dx
                right_slope = (U_ext[var, i+1] - U_ext[var, i]) / dx
                slopes[var, i] = self.get_limiter(left_slope, right_slope)
        for i in range(n-1):
            for var in range(3):
                UL[var, i] = U_ext[var, i] + 0.5 * dx * slopes[var, i]
                UR[var, i] = U_ext[var, i+1] - 0.5 * dx * slopes[var, i+1]
            UL[0, i] = max(UL[0, i], self.rho_min)
            UR[0, i] = max(UR[0, i], self.rho_min)
        return UL, UR

    def solve(self, U, dx, dt):
        """Godunov's method with MUSCL scheme and boundary conditions."""
        n = len(U[0])
        flux = np.zeros((3, n + 1))
        U_new = np.zeros_like(U)
        U_ext = self.apply_boundary_conditions(U)
        UL, UR = self.muscl_reconstruction(U_ext, dx)
        for i in range(n + 1):
            if self.flux_method == 'HLLC':
                flux[:, i] = self.hllc_flux(UL[:, i + 1], UR[:, i + 1])
            else:
                raise ValueError("Unsupported flux method")
        for i in range(n):
            U_new[:, i] = U[:, i] - dt / dx * (flux[:, i + 1] - flux[:, i])
        return U_new

    def plot_density_evolution(self, densities_snapshots, times, x):
        """Plot density profiles at multiple time steps."""
        plt.figure(figsize=(10, 6))
        for t, rho in zip(times, density_snapshots):
            plt.plot(x, rho, label=f't = {t:.3f}', alpha=0.7)
        plt.title('Density Evolution (Sod Shock Tube)')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.savefig('density_evolution.png')

# Example usage
if __name__ == "__main__":
    # Setup
    n_cells = 100
    dx = 1.0 / n_cells
    gamma = 1.4
    cfl = 0.5
    t_final = 0.1
    snapshot_interval = 0.02  # Save density every 0.02s

    # Initial conditions (Sod shock tube)
    U = np.zeros((3, n_cells))
    x = np.linspace(0, 1, n_cells + 1)
    xmid = (x[1:] + x[:-1]) / 2.0
    for i in range(n_cells):
        if xmid[i] < 0.5:
            U[:, i] = np.array([1.0, 0.0, 1.0 / (gamma - 1)])
        else:
            U[:, i] = np.array([0.125, 0.0, 0.1 / (gamma - 1)])

    solver = GodunovSolver(gamma=gamma, flux_method='HLLC', limiter='minmod', cfl=cfl, bc_type='transmissive')

    # Time stepping with snapshots
    t = 0
    density_snapshots = [U[0].copy()]  # Initial density
    times = [0.0]
    next_snapshot = snapshot_interval
    while t < t_final:
        dt = solver.compute_cfl_dt(U, dx)
        if t + dt > t_final:
            dt = t_final - t
        U = solver.solve(U, dx, dt)
        t += dt
        if t >= next_snapshot:
            density_snapshots.append(U[0].copy())
            times.append(t)
            next_snapshot += snapshot_interval

    # Plot density evolution
    solver.plot_density_evolution(density_snapshots, times, xmid)
    print("Final density:", U[0])
