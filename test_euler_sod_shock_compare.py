# read sedov.dat and solution.dat files
import numpy as np
import matplotlib.pyplot as plt

def read_solution(filename):
    """Reads the solution.dat file and returns a list of (time, x, W) tuples."""
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('# Step'):
            # Parse time
            time_line = line
            time = float(time_line.split('Time')[1].strip())
            # Skip header
            i += 2
            x_list, w_list = [], []
            while i < len(lines) and not lines[i].startswith('#'):
                vals = lines[i].split()
                if len(vals) == 4:
                    x_list.append(float(vals[0]))
                    w_list.append([float(vals[1]), float(vals[2]), float(vals[3])])
                i += 1
            x = np.array(x_list)
            W = np.array(w_list).T  # shape: (3, N)
            data.append((time, x, W))
        else:
            i += 1
    return data

def sod_analytic(x, t, gamma=1.4, 
                left_state=(1.0, 0.0, 1.0), 
                right_state=(0.125, 0.0, 0.1), 
                x0=0.5):
    """
    Analytic solution for the Sod shock tube problem.
    Returns density, velocity, pressure arrays at positions x and time t.
    """
    # Unpack states
    rho_l, u_l, p_l = left_state
    rho_r, u_r, p_r = right_state

    # Compute sound speeds
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    # Helper functions for pressure-velocity relations
    def fL(p):
        if p > p_l:
            A = np.sqrt((gamma + 1) / (2 * gamma) * p / p_l + (gamma - 1) / (2 * gamma))
            return (p - p_l) / (rho_l * c_l * A)
        else:
            return (2 * c_l / (gamma - 1)) * ((p / p_l) ** ((gamma - 1) / (2 * gamma)) - 1)

    def fR(p):
        if p > p_r:
            A = np.sqrt((gamma + 1) / (2 * gamma) * p / p_r + (gamma - 1) / (2 * gamma))
            return (p - p_r) / (rho_r * c_r * A)
        else:
            return (2 * c_r / (gamma - 1)) * ((p / p_r) ** ((gamma - 1) / (2 * gamma)) - 1)

    # Solve for p_star using Newton-Raphson
    def find_p_star():
        p0 = 0.5 * (p_l + p_r)
        p = p0
        for _ in range(50):
            f = fL(p) + fR(p) + u_r - u_l
            # Numerical derivative
            dp = 1e-6 * p
            fprime = (fL(p + dp) + fR(p + dp) + u_r - u_l - f) / dp
            p_new = p - f / fprime
            if abs(p_new - p) < 1e-8:
                break
            p = p_new
        return p

    p_star = find_p_star()
    u_star = 0.5 * (u_l + u_r) + 0.5 * (fR(p_star) - fL(p_star))

    # Compute densities in star region
    if p_star > p_l:
        rho_star_l = rho_l * ((p_star / p_l + (gamma - 1) / (gamma + 1)) /
                              ((gamma - 1) / (gamma + 1) * p_star / p_l + 1))
    else:
        rho_star_l = rho_l * (p_star / p_l) ** (1 / gamma)
    if p_star > p_r:
        rho_star_r = rho_r * ((p_star / p_r + (gamma - 1) / (gamma + 1)) /
                              ((gamma - 1) / (gamma + 1) * p_star / p_r + 1))
    else:
        rho_star_r = rho_r * (p_star / p_r) ** (1 / gamma)

    # Compute wave speeds
    if p_star > p_l:
        s_l = u_l - c_l * np.sqrt((gamma + 1) / (2 * gamma) * p_star / p_l + (gamma - 1) / (2 * gamma))
    else:
        s_l = u_l - c_l
    if p_star > p_r:
        s_r = u_r + c_r * np.sqrt((gamma + 1) / (2 * gamma) * p_star / p_r + (gamma - 1) / (2 * gamma))
    else:
        s_r = u_r + c_r

    # Head and tail of rarefaction
    shl = u_l - c_l
    stl = u_star - np.sqrt(gamma * p_star / rho_star_l)
    shr = u_r + c_r
    str_ = u_star + np.sqrt(gamma * p_star / rho_star_r)

    xi = (x - x0) / t
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)

    for i in range(len(x)):
        if xi[i] < shl:
            # Left data
            rho[i] = rho_l
            u[i] = u_l
            p[i] = p_l
        elif shl <= xi[i] < stl:
            # Left rarefaction fan
            u[i] = (2 / (gamma + 1)) * (c_l + (gamma - 1) / 2 * u_l + xi[i])
            c = (2 / (gamma + 1)) * (c_l + (gamma - 1) / 2 * (u_l - xi[i]))
            rho[i] = rho_l * (c / c_l) ** (2 / (gamma - 1))
            p[i] = p_l * (c / c_l) ** (2 * gamma / (gamma - 1))
        elif stl <= xi[i] < u_star:
            # Left star region
            rho[i] = rho_star_l
            u[i] = u_star
            p[i] = p_star
        elif u_star <= xi[i] < s_r:
            # Right star region
            rho[i] = rho_star_r
            u[i] = u_star
            p[i] = p_star
        else:
            # Right data
            rho[i] = rho_r
            u[i] = u_r
            p[i] = p_r

    return rho, u, p

def compare_sod_analytic_to_solution(solution_file, x0=0.5):
    """
    Compare the analytic Sod shock tube solution to the numerical solution.

    Parameters
    ----------
    solution_file : str
        Path to the solution.dat file containing the numerical results.
    x0 : float, optional
        The initial position of the discontinuity (shock interface) in the domain.
        By default, x0=0.5, which means the initial left/right states are split at x=0.5.

    This function reads the numerical solution, computes the analytic solution at the same
    positions and time, and plots both for visual comparison.
    """
    # Read the numerical solution from file
    data = read_solution(solution_file)
    # Use the last time snapshot for comparison
    time, x, W = data[-1]
    rho_num, u_num, p_num = W[0], W[1], W[2]

    # Compute analytic solution at the same positions and time
    # x0 is the initial location of the discontinuity between left and right states
    rho_ana, u_ana, p_ana = sod_analytic(x, time, x0=x0)

    # Plot density
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(x, rho_num, 'b-', label='Numerical')
    plt.plot(x, rho_ana, 'r--', label='Analytic')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Plot velocity
    plt.subplot(3, 1, 2)
    plt.plot(x, u_num, 'b-', label='Numerical')
    plt.plot(x, u_ana, 'r--', label='Analytic')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)

    # Plot pressure
    plt.subplot(3, 1, 3)
    plt.plot(x, p_num, 'b-', label='Numerical')
    plt.plot(x, p_ana, 'r--', label='Analytic')
    plt.xlabel('x')
    plt.ylabel('Pressure')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Sod Shock Tube: Numerical vs Analytic at t={time:.3f}')
    plt.tight_layout()
    plt.show()

def main():
    solution_file = 'solution.dat'  # Path to your solution.dat file
    compare_sod_analytic_to_solution(solution_file, x0=0.5)

if __name__ == '__main__':
    main()
