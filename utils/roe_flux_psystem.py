import numpy as np

def compute_roe_flux(u_L, u_R, flux_func, epsilon=1e-6, delta=1e-6):
    """
    Compute Roe flux for a general 1D hyperbolic conservation law.
    
    Parameters:
    u_L, u_R : ndarray, left and right state vectors
    flux_func : function, computes flux f(u)
    epsilon : float, perturbation for finite difference Jacobian
    delta : float, entropy fix parameter
    
    Returns:
    F : ndarray, Roe flux at interface
    """
    m = len(u_L)  # Number of components
    u_avg = 0.5 * (u_L + u_R)  # Simple average for Jacobian evaluation
    
    # Numerical Jacobian
    A = np.zeros((m, m))
    for j in range(m):
        u_plus = u_avg.copy()
        u_minus = u_avg.copy()
        u_plus[j] += epsilon
        u_minus[j] -= epsilon
        A[:, j] = (flux_func(u_plus) - flux_func(u_minus)) / (2 * epsilon)
    
    # Eigenstructure
    eigenvalues, R = np.linalg.eig(A)
    L = np.linalg.inv(R)  # Left eigenvectors
    
    # Wave strengths
    delta_u = u_R - u_L
    alpha = L @ delta_u
    
    # Entropy fix
    abs_eigenvalues = np.array([max(abs(lam), delta) for lam in eigenvalues])
    
    # Roe flux
    F_avg = 0.5 * (flux_func(u_L) + flux_func(u_R))
    correction = 0.5 * sum(abs_eigenvalues[k] * alpha[k] * R[:, k] for k in range(m))
    F = F_avg - correction
    
    return F

# p-System flux function
def p_system_flux(u, sigma_func=lambda v: v**2):
    """
    Flux function for p-System: u = [v, w], f = [w, p(v)], p(v) = -sigma(v)
    """
    v, w = u
    return np.array([w, -sigma_func(v)])

# Example usage for p-System
def main():
    # Example states
    u_L = np.array([1.0, 0.5])  # [v_L, w_L]
    u_R = np.array([0.8, 0.3])  # [v_R, w_R]
    
    # Compute Roe flux
    F = compute_roe_flux(u_L, u_R, p_system_flux)
    
    print("Left state u_L:", u_L)
    print("Right state u_R:", u_R)
    print("Roe flux F:", F)

if __name__ == "__main__":
    main()