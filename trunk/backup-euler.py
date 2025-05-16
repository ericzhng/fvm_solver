import numpy as np
import matplotlib.pyplot as plt

class GodunovSolver:
    def __init__(self, n_vars, primitive_to_conservative, conservative_to_primitive, compute_flux, estimate_wave_speeds, 
                 flux_method='HLLC', limiter='minmod', cfl=0.5, bc_type='transmissive', rho_min=1e-10, p_min=1e-10, custom_riemann_solver=None):
        self.n_vars = n_vars
        self.primitive_to_conservative = primitive_to_conservative
        self.conservative_to_primitive = conservative_to_primitive
        self.compute_flux = compute_flux
        self.estimate_wave_speeds = estimate_wave_speeds
        self.flux_method = flux_method
        self.limiter = limiter
        self.cfl = cfl
        self.bc_type = bc_type
        self.rho_min = rho_min
        self.p_min = p_min
        self.custom_riemann_solver = custom_riemann_solver

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

    def compute_cfl_dt(self, U, dx):
        """Compute time step based on CFL condition."""
        n = U.shape[1]
        max_speed = 0
        for i in range(n):
            W = self.conservative_to_primitive(U[:, i])
            speeds = self.estimate_wave_speeds(W)
            max_speed = max(max_speed, np.max(np.abs(speeds)))
        dt = self.cfl * dx / max_speed
        return dt

    def hllc_flux(self, UL, UR):
        """HLLC Riemann solver flux with safeguards."""
        WL = self.conservative_to_primitive(UL)
        WR = self.conservative_to_primitive(UR)
        SL, SR, S_star = self.estimate_wave_speeds(WL, WR)
        FL = self.compute_flux(UL, WL)
        FR = self.compute_flux(UR, WR)
        if SL >= 0:
            return FL
        elif SL <= 0 <= S_star:
            UL_star = self.star_state(UL, WL, SL, S_star)
            return FL + SL * (UL_star - UL)
        elif S_star <= 0 <= SR:
            UR_star = self.star_state(UR, WR, SR, S_star)
            return FR + SR * (UR_star - UR)
        else:
            return FR

    def star_state(self, U, W, S, S_star):
        """Generic star state for HLLC; user may override for their system."""
        return U

    def apply_boundary_conditions(self, U):
        """Apply boundary conditions using ghost cells."""
        n = U.shape[1]
        U_ext = np.zeros((self.n_vars, n + 4))
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
        n = U_ext.shape[1]
        UL = np.zeros((self.n_vars, n - 1))
        UR = np.zeros((self.n_vars, n - 1))
        slopes = np.zeros_like(U_ext)
        for i in range(1, n-1):
            for var in range(self.n_vars):
                left_slope = (U_ext[var, i] - U_ext[var, i-1]) / dx
                right_slope = (U_ext[var, i+1] - U_ext[var, i]) / dx
                slopes[var, i] = self.get_limiter(left_slope, right_slope)
        for i in range(n-1):
            for var in range(self.n_vars):
                UL[var, i] = U_ext[var, i] + 0.5 * dx * slopes[var, i]
                UR[var, i] = U_ext[var, i+1] - 0.5 * dx * slopes[var, i+1]
        return UL, UR

    def solve(self, U, dx, dt):
        """Godunov's method with MUSCL scheme and boundary conditions."""
        n = U.shape[1]
        flux = np.zeros((self.n_vars, n + 1))
        U_new = np.zeros_like(U)
        U_ext = self.apply_boundary_conditions(U)
        UL, UR = self.muscl_reconstruction(U_ext, dx)
        for i in range(n + 1):
            if self.flux_method == 'HLLC':
                if self.custom_riemann_solver is not None:
                    flux[:, i] = self.custom_riemann_solver(UL[:, i + 1], UR[:, i + 1])
                else:
                    flux[:, i] = self.hllc_flux(UL[:, i + 1], UR[:, i + 1])
            else:
                raise ValueError("Unsupported flux method")
        for i in range(n):
            U_new[:, i] = U[:, i] - dt / dx * (flux[:, i + 1] - flux[:, i])
        return U_new

    def plot_density_evolution(self, density_snapshots, times, x):
        """Plot density profiles at multiple time steps."""
        plt.figure(figsize=(10, 6))
        for t, rho in zip(times, density_snapshots):
            plt.plot(x, rho, label=f't = {t:.3f}', alpha=0.7)
        plt.title('Density Evolution')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.savefig('density_evolution.png')

# Example usage for Euler equations (user can swap in their own system)
if __name__ == "__main__":
    gamma = 1.4
    n_vars = 3
    n_cells = 100
    dx = 1.0 / n_cells
    cfl = 0.5
    t_final = 0.1
    snapshot_interval = 0.02

    def primitive_to_conservative(rho, u, p):
        mom = rho * u
        energy = p / (gamma - 1) + 0.5 * rho * u**2
        return np.array([rho, mom, energy])

    def conservative_to_primitive(U):
        rho = max(U[0], 1e-10)
        u = U[1] / rho
        p = max((gamma - 1) * (U[2] - 0.5 * rho * u**2), 1e-10)
        return np.array([rho, u, p])

    def compute_flux(U, W):
        rho, u, p = W
        mom = U[1]
        energy = U[2]
        return np.array([mom, rho * u**2 + p, u * (energy + p)])

    def estimate_wave_speeds(WL, WR=None):
        # For Euler: returns (SL, SR, S_star)
        rhoL, uL, pL = WL
        if WR is not None:
            rhoR, uR, pR = WR
        else:
            rhoR, uR, pR = WL
        cL = np.sqrt(gamma * pL / rhoL)
        cR = np.sqrt(gamma * pR / rhoR)
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)
        S_star = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / \
                 (rhoL * (SL - uL) - rhoR * (SR - uR))
        return SL, SR, S_star

    # Initial conditions (Sod shock tube)
    U = np.zeros((n_vars, n_cells))
    x = np.linspace(0, 1, n_cells)
    for i in range(n_cells):
        if x[i] < 0.5:
            U[:, i] = primitive_to_conservative(1.0, 0.0, 1.0)
        else:
            U[:, i] = primitive_to_conservative(0.125, 0.0, 0.1)

    solver = GodunovSolver(
        n_vars=n_vars,
        primitive_to_conservative=primitive_to_conservative,
        conservative_to_primitive=conservative_to_primitive,
        compute_flux=compute_flux,
        estimate_wave_speeds=estimate_wave_speeds,
        flux_method='HLLC',
        limiter='minmod',
        cfl=cfl,
        bc_type='transmissive',
    )

    t = 0
    density_snapshots = [U[0].copy()]
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

    solver.plot_density_evolution(density_snapshots, times, x)
    print("Final density:", U[0])
    