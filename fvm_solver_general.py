import math

# Problem definition (example: linear advection u_t + a u_x = 0)
class Problem:
    def __init__(self, a=1.0):
        self.a = a  # Advection speed

    def flux(self, u):
        """Compute flux f(u)."""
        return self.a * u

    def max_wave_speed(self, u):
        """Estimate maximum wave speed for CFL condition."""
        return abs(self.a)

# General FVM Solver
class FVMSolver:
    def __init__(self, problem, N, dx, CFL, T):
        self.problem = problem
        self.N = N  # Number of cells
        self.dx = dx  # Cell width
        self.CFL = CFL  # CFL number
        self.T = T  # Final time
        self.x = [i * dx - 0.5 for i in range(N)]  # Cell centers

    # Minmod limiter
    def minmod(self, a, b):
        if a * b > 0:
            return math.copysign(min(abs(a), abs(b)), a)
        return 0.0

    # Flux computation
    def compute_flux(self, U, method="upwind"):
        F = [0.0] * self.N
        if method == "upwind":
            for i in range(self.N):
                F[i] = self.problem.flux(U[i])  # F_{i+1/2} = f(U_i)
        elif method == "linear":
            # Centered slope (unlimited)
            sigma = [(U[(i+1)%self.N] - U[(i-1)%self.N]) / (2 * self.dx) for i in range(self.N)]
            for i in range(self.N):
                U_left = U[i] + sigma[i] * (self.dx / 2)  # U_{i+1/2}^L
                F[i] = self.problem.flux(U_left)
        elif method == "limited":
            # Minmod limited slope
            sigma = [self.minmod((U[i] - U[(i-1)%self.N]) / self.dx, 
                                (U[(i+1)%self.N] - U[i]) / self.dx) for i in range(self.N)]
            for i in range(self.N):
                U_left = U[i] + sigma[i] * (self.dx / 2)  # U_{i+1/2}^L
                F[i] = self.problem.flux(U_left)
        elif method == "muscl":
            # MUSCL with minmod limiter
            sigma = [self.minmod((U[i] - U[(i-1)%self.N]) / self.dx, 
                                (U[(i+1)%self.N] - U[i]) / self.dx) for i in range(self.N)]
            for i in range(self.N):
                U_i_plus_half_L = U[i] + sigma[i] * (self.dx / 2)  # Left state at i+1/2
                F[i] = self.problem.flux(U_i_plus_half_L)
        return F

    # Solve
    def solve(self, U0, method="muscl"):
        U = U0.copy()
        t = 0.0
        while t < self.T:
            # Compute time step based on CFL
            max_speed = max(abs(self.problem.max_wave_speed(U[i])) for i in range(self.N))
            dt = self.CFL * self.dx / max_speed if max_speed > 0 else self.T - t
            
            # Compute fluxes
            F = self.compute_flux(U, method)
            
            # Update cell averages
            U_new = U.copy()
            for i in range(self.N):
                U_new[i] = U[i] - dt/self.dx * (F[i] - F[(i-1)%self.N])
            
            U = U_new
            t += min(dt, self.T - t)
        return U

# Example: Linear advection
problem = Problem(a=1.0)
N = 100
dx = 1.0 / N
CFL = 0.8
T = 1.0

# Initial condition
x = [i * dx - 0.5 for i in range(N)]
U0 = [math.sin(2 * math.pi * xi) for xi in x]  # u(x,0) = sin(2πx)

# Run solver
solver = FVMSolver(problem, N, dx, CFL, T)
U = solver.solve(U0, method="muscl")

# Output final U (for comparison)
print(U)