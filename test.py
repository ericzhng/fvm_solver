import numpy as np
from solver import GodunovSolver
from equation import ShallowWaterSystem, EulerEquationSystem

def run_test(equation_system, test_name, n_cells=100, t_final=0.1, snapshot_interval=0.02):
    dx = 1.0 / n_cells
    cfl = 0.5
    x = np.linspace(0, 1, n_cells)
    if isinstance(equation_system, ShallowWaterSystem):
        U = np.zeros((2, n_cells))
        for i in range(n_cells):
            W = np.array([1.0, 0.0]) if x[i] < 0.5 else np.array([0.1, 0.0])
            U[:, i] = equation_system.to_conservative(W)
    else:  # EulerEquationSystem
        U = np.zeros((3, n_cells))
        for i in range(n_cells):
            W = np.array([1.0, 0.0, 1.0]) if x[i] < 0.5 else np.array([0.125, 0.0, 0.1])
            U[:, i] = equation_system.to_conservative(W)

    reconstruction_methods = [
        ('piecewise_constant', 'minmod'),
        ('muscl', 'minmod'),
        ('muscl', 'superbee'),
        ('muscl', 'van_leer'),
        ('ppm', 'minmod'),
        ('weno5', 'minmod')
    ]
    for flux_method in ['HLLC', 'Roe']:  # Limit to HLLC and Roe for brevity
        for recon_method, limiter in reconstruction_methods:
            solver = GodunovSolver(equation_system=equation_system, flux_method=flux_method, limiter=limiter, reconstruction_method=recon_method, cfl=cfl, bc_type='transmissive')
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
            print(f"Final {equation_system.get_variable_names()[0].lower()} ({test_name}, {recon_method}, {limiter}, {flux_method}):", U_current[0])

if __name__ == "__main__":
    shallow_water = ShallowWaterSystem(g=9.81, h_min=1e-10)
    run_test(shallow_water, "Dam-Break")
    euler = EulerEquationSystem(gamma=1.4, rho_min=1e-10, p_min=1e-10)
    run_test(euler, "Sod Shock Tube")