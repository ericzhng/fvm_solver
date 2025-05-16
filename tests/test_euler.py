import numpy as np
from src.equation import EulerEquationSystem
from src.solver import GodunovSolver

def run_euler_test(n_cells: int = 100, t_final: float = 0.1, snapshot_interval: float = 0.02):
    """Run Sod shock tube test for Euler equations.
    
    Args:
        n_cells: Number of grid cells
        t_final: Final simulation time
        snapshot_interval: Time interval for snapshots
    """
    dx = 1.0 / n_cells
    x = np.linspace(0, 1, n_cells)
    equation_system = EulerEquationSystem(gamma=1.4, rho_min=1e-10, p_min=1e-10)
    
    # Initialize Sod shock tube
    U = np.zeros((3, n_cells))
    for i in range(n_cells):
        W = np.array([1.0, 0.0, 1.0]) if x[i] < 0.5 else np.array([0.125, 0.0, 0.1])
        U[:, i] = equation_system.to_conservative(W)
    
    # Test configurations
    configs = [
        ('piecewise_constant', 'minmod'),
        ('muscl', 'minmod'),
        ('muscl', 'superbee'),
        ('muscl', 'van_leer'),
        ('muscl', 'mc'),
        ('muscl', 'koren'),
        ('muscl', 'osher'),
        ('muscl', 'sweby'),
        ('muscl', 'umist'),
        ('ppm', 'minmod'),
        ('weno5', 'minmod')
    ]
    
    for flux_method in ['HLLC', 'Roe']:
        for recon_method, limiter in configs:
            solver = GodunovSolver(
                equation_system=equation_system,
                flux_method=flux_method,
                limiter=limiter,
                reconstruction_method=recon_method,
                cfl=0.5,
                bc_type='transmissive'
            )
            t = 0
            U_current = U.copy()
            snapshots = [U_current[0].copy()]
            times = [0.0]
            next_snapshot = snapshot_interval
            
            while t < t_final:
                dt = solver.compute_cfl_dt(U_current, dx)
                if t + dt > t_final:
                    dt = t_final - t
                U_current = solver.solve(U_current, dx, dt)
                t += dt
                if t >= next_snapshot:
                    snapshots.append(U_current[0].copy())
                    times.append(t)
                    next_snapshot += snapshot_interval
            
            solver.plot_variable_evolution(snapshots, times, x)
            print(f"Final density (Sod Shock Tube, {recon_method}, {limiter}, {flux_method}):", U_current[0])

if __name__ == "__main__":
    run_euler_test()