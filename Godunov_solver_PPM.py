import numpy as np
import matplotlib.pyplot as plt

# Physical constant for adiabatic index
gamma = 1.4

# Apply safeguards to ensure physical density and pressure
def apply_safeguards(rho, p):
    rho_min = 1e-10  # Minimum allowable density
    p_min = 1e-10    # Minimum allowable pressure
    rho = np.maximum(rho, rho_min)
    p = np.maximum(p, p_min)
    return rho, p

# Perform PPM reconstruction for left and right states
def ppm_reconstruction(q, nx):
    dq = np.zeros_like(q)  # Slope array
    q_l = np.zeros_like(q)  # Left states
    q_r = np.zeros_like(q)  # Right states
    
    for i in range(2, nx-2):
        # Compute limited slopes for monotonicity
        dq_m = q[:, i] - q[:, i-1]  # Backward difference
        dq_p = q[:, i+1] - q[:, i]  # Forward difference
        dq_c = 0.5 * (q[:, i+1] - q[:, i-1])  # Central difference
        dq[:, i] = np.minimum(np.abs(dq_c), 2 * np.minimum(np.abs(dq_m), np.abs(dq_p))) * np.sign(dq_c)
        # Reconstruct left and right states using parabolic method
        q_l[:, i] = q[:, i] - 0.5 * dq[:, i] + (1.0/6.0) * (dq[:, i-1] - 2 * dq[:, i] + dq[:, i+1])
        q_r[:, i] = q[:, i] + 0.5 * dq[:, i] - (1.0/6.0) * (dq[:, i-1] - 2 * dq[:, i] + dq[:, i+1])
    
    return q_l, q_r

# Convert primitive variables to conserved variables
def primitive_to_conserved(rho, u, p):
    rho, p = apply_safeguards(rho, p)  # Apply safeguards
    mom = rho * u  # Momentum
    e = p / (gamma - 1) + 0.5 * rho * u**2  # Total energy
    return np.array([rho, mom, e])

# Convert conserved variables to primitive variables
def conserved_to_primitive(q):
    rho_min = 1e-10  # Minimum allowable density
    rho = q[0]  # Density
    rho = np.maximum(rho, rho_min)
    u = q[1] / rho   # Velocity, avoid div by zero
    p = (gamma - 1) * (q[2] - 0.5 * rho * u**2)  # Pressure
    rho, p = apply_safeguards(rho, p)  # Apply safeguards
    return rho, u, p

# Compute Euler fluxes from conserved variables
def euler_flux(q):
    rho, u, p = conserved_to_primitive(q)
    f = np.zeros_like(q)
    f[0] = rho * u  # Mass flux
    f[1] = rho * u**2 + p  # Momentum flux
    f[2] = u * (q[2] + p)  # Energy flux
    return f

# Compute Godunov flux using a simple Riemann solver
def godunov_flux(q_l, q_r):
    # Extract primitive variables
    rho_l, u_l, p_l = conserved_to_primitive(q_l)
    rho_r, u_r, p_r = conserved_to_primitive(q_r)
    
    # Compute sound speeds
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)
    
    # Estimate wave speeds for Riemann problem
    s_l = min(u_l - c_l, u_r - c_r)  # Left wave speed
    s_r = max(u_l + c_l, u_r + c_r)  # Right wave speed
    
    # Compute fluxes at left and right states
    f_l = euler_flux(q_l)
    f_r = euler_flux(q_r)
    
    # Apply Godunov's upwind method
    if s_l >= 0:
        return f_l
    elif s_r <= 0:
        return f_r
    else:
        # Compute intermediate state flux, handle small denominator
        denom = s_r - s_l
        return np.where(denom > 1e-8, 
                        (s_r * f_l - s_l * f_r + s_l * s_r * (q_r - q_l)) / denom, 
                        0.5 * (f_l + f_r))

# Set up initial conditions for Sod shock tube
def sod_initial_conditions(nx, x):
    q = np.zeros((3, nx))  # Array for conserved variables
    for i in range(nx):
        if x[i] < 0.5:
            # Left state: high pressure and density
            q[:, i] = primitive_to_conserved(1.0, 0.0, 1.0)
        else:
            # Right state: low pressure and density
            q[:, i] = primitive_to_conserved(0.125, 0.0, 0.1)
    return q

# Main Godunov solver for 1D Euler equations
def godunov_solver(nx, dx, t_final, dt):
    # Initialize grid and solution
    x = np.linspace(0, 1, nx)
    q = sod_initial_conditions(nx, x)
    t = 0.0
    
    while t < t_final:
        # Compute CFL-compliant time step
        rho, u, p = conserved_to_primitive(q)
        c = np.sqrt(gamma * p / rho)
        dt = min(dt, 0.5 * dx / np.max(np.abs(u) + c))
        
        # Reconstruct left and right states with PPM
        q_l, q_r = ppm_reconstruction(q, nx)
        
        # Compute fluxes at cell interfaces
        flux = np.zeros_like(q)
        for i in range(1, nx-1):
            flux[:, i] = godunov_flux(q_r[:, i-1], q_l[:, i])
        
        # Update conserved variables using finite volume method
        q[:, 1:-1] = q[:, 1:-1] - (dt / dx) * (flux[:, 1:-1] - flux[:, :-2])
        
        # Apply transmissive boundary conditions
        q[:, 0] = q[:, 1]
        q[:, -1] = q[:, -2]
        
        # Ensure physical values after update
        rho, u, p = conserved_to_primitive(q)
        q = primitive_to_conserved(rho, u, p)
        
        t += dt
        # show progress
        print(f"Time: {t:.3f}, Max Density: {np.max(rho):.3f}, Max Pressure: {np.max(p):.3f}")
    
    return x, q

# Define simulation parameters
nx = 100  # Number of grid points
dx = 1.0 / (nx - 1)  # Spatial step size
t_final = 0.2  # Final simulation time
dt = 0.001  # Initial time step

# Execute solver
x, q = godunov_solver(nx, dx, t_final, dt)

# Plot density, velocity, and pressure
rho, u, p = conserved_to_primitive(q)
plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.plot(x, rho, label='Density')
plt.legend()
plt.subplot(312)
plt.plot(x, u, label='Velocity')
plt.legend()
plt.subplot(313)
plt.plot(x, p, label='Pressure')
plt.legend()
plt.tight_layout()
plt.show()