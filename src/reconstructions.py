import numpy as np
from .equation import EquationSystem
from .limiters import Limiter


class Reconstruction:
    """Class to perform spatial reconstructions for Godunov-type schemes.

    Supports piecewise constant, MUSCL, PPM, and WENO5 methods.

    Ghost Cells: n_ghost=2 suffices for MUSCL, WENO5, and simple PPM; n_ghost=3 is preferred for full PPM to support the 5-point stencil and 4-point boundary extrapolation.
    Boundary Conditions: The solver must handle all BCs, including velocity flipping for reflective BCs in apply_bcs.
    Accuracy: Boundary extrapolation is second-order for all methods, aligning with MUSCL and simple PPM; full PPM interior is third-order.
    
    Args:
        equation_system (EquationSystem): System defining primitive/conservative conversions.
        limiter (str, optional): Slope limiter type (e.g., 'minmod'). Defaults to None.
    """

    def __init__(self, equation_system: EquationSystem, limiter: str = None):
        """Initialize the reconstruction scheme.

        Args:
            equation_system (EquationSystem): The equation system for state conversions.
            limiter (str, optional): Slope limiter for MUSCL ('minmod', 'superbee', 'vanleer', 'none').
        """
        self.equation_system = equation_system
        self.limiter = Limiter(limiter) if limiter else None

    def _to_primitive_array(self, U: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells + 2*n_ghost).

        Returns:
            np.ndarray: Primitive variables, shape (n_vars, n_cells + 2*n_ghost).
        """
        n_cells = U.shape[1]
        return np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n_cells)]).T

    def _reconstruct_states(self, W_L: np.ndarray, W_R: np.ndarray, n_cells: int) -> tuple:
        """Convert primitive interface values to conservative states.

        Args:
            W_L (np.ndarray): Left primitive values, shape (n_vars, n_cells - 1).
            W_R (np.ndarray): Right primitive values, shape (n_vars, n_cells - 1).
            n_cells (int): Number of physical cells.

        Returns:
            tuple: (UL, UR), conservative states, each shape (n_vars, n_cells - 1).
        """
        UL = np.zeros((W_L.shape[0], n_cells - 1))
        UR = np.zeros((W_R.shape[0], n_cells - 1))
        for i in range(n_cells - 1):
            UL[:, i] = self.equation_system.to_conservative(W_L[:, i])
            UR[:, i] = self.equation_system.to_conservative(W_R[:, i])
        return UL, UR

    def _linear_extrapolation(self, W: np.ndarray, i: int, j: int, n_cells: int, n_ghost: int) -> tuple:
        """Compute boundary interface values using linear extrapolation.

        Args:
            W (np.ndarray): Primitive variables, shape (n_vars, n_cells + 2*n_ghost).
            i (int): Interface index (0 to n_cells - 2).
            j (int): Variable index.
            n_cells (int): Number of physical cells.
            n_ghost (int): Number of ghost cells per side.

        Returns:
            tuple: (W_L, W_R), left and right interface values for variable j.
        """
        if i < n_ghost:  # Left boundary
            idx = n_ghost + i + 1           # Use cells i=n_ghost+1, n_ghost+2, ...
            W_L = W[j, idx] - 0.5 * (W[j, idx + 1] - W[j, idx])
            W_R = W[j, idx + 1] + 0.5 * (W[j, idx + 1] - W[j, idx])
        else:  # Right boundary
            idx = n_cells + n_ghost - (1 - i) - 1         # Use cells n_cells+n_ghost-2, ...
            W_L = W[j, idx] + 0.5 * (W[j, idx] - W[j, idx - 1])
            W_R = W[j, idx + 1] - 0.5 * (W[j, idx + 1] - W[j, idx])
        return W_L, W_R

    # reconstruction methods
    # --------------------------
    # piecewise constant
    # --------------------------
    def piecewise_constant(self, U: np.ndarray, dx: float) -> tuple:
        """Perform piecewise constant reconstruction.

        UL[i] = U[i], UR[i] = U[i+1]

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            dx (float): Spatial grid spacing.

        Returns:
            tuple: Left and right reconstructed states (UL, UR).
        """
        UL = U[:, :-1].copy()
        UR = U[:, 1:].copy()
        return UL, UR

    # reconstruction methods
    # --------------------------
    # MUSCL
    # Slope Limiting: minmod, superbee, vanleer, none
    # --------------------------
    def muscl(self, U: np.ndarray, dx: float) -> tuple:
        """Perform MUSCL reconstruction with slope limiting.

        The Monotonic Upwind Scheme for Conservation Laws
        The scheme achieves:
            Second-order accuracy in smooth regions by using linear reconstruction.
            Monotonicity via slope limiters to prevent oscillations near shocks.
            Total Variation Diminishing (TVD) properties under certain conditions.

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            dx (float): Spatial grid spacing.

        Returns:
            tuple: Left and right reconstructed states (UL, UR).
        """
        n_vars, n_cells = U.shape
        UL = np.zeros((n_vars, n_cells - 1))
        UR = np.zeros((n_vars, n_cells - 1))
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n_cells)]).T

        for i in range(n_cells - 1):
            for j in range(n_vars):
                if 2 <= i <= n_cells - 3:  # Interior cells: standard MUSCL
                    sigma_L = self.limiter.limit(
                        (W[j, i] - W[j, i - 1]) / dx,
                        (W[j, i - 1] - W[j, i - 2]) / dx
                    ) if self.limiter else (W[j, i] - W[j, i - 1]) / dx
                    sigma_R = self.limiter.limit(
                        (W[j, i + 1] - W[j, i]) / dx,
                        (W[j, i] - W[j, i - 1]) / dx
                    ) if self.limiter else (W[j, i + 1] - W[j, i]) / dx
                    W_L = W[j, i] - 0.5 * dx * sigma_L
                    W_R = W[j, i] + 0.5 * dx * sigma_R
                else:  # Boundary cells: linear extrapolation
                    if i == 0:  # Left boundary (x_{1/2})
                        W_L = W[j, 1] - 0.5 * (W[j, 2] - W[j, 1])
                        W_R = W[j, 2] + 0.5 * (W[j, 2] - W[j, 1])
                    elif i == 1:  # Near left (x_{3/2})
                        W_L = W[j, 2] - 0.5 * (W[j, 3] - W[j, 2])
                        W_R = W[j, 3] + 0.5 * (W[j, 3] - W[j, 2])
                    elif i == n_cells - 2:  # Right boundary (x_{n_cells-1/2})
                        W_L = W[j, n_cells - 2] + 0.5 * (W[j, n_cells - 2] - W[j, n_cells - 3])
                        W_R = W[j, n_cells - 1] - 0.5 * (W[j, n_cells - 1] - W[j, n_cells - 2])
                    elif i == n_cells - 3:  # Near right (x_{n_cells-3/2})
                        W_L = W[j, n_cells - 3] + 0.5 * (W[j, n_cells - 3] - W[j, n_cells - 4])
                        W_R = W[j, n_cells - 2] - 0.5 * (W[j, n_cells - 2] - W[j, n_cells - 3])

                W_tmp_L = W[:, i].copy()
                W_tmp_R = W[:, i + 1].copy()
                W_tmp_L[j] = W_L
                W_tmp_R[j] = W_R
                UL[:, i] = self.equation_system.to_conservative(W_tmp_L)
                UR[:, i] = self.equation_system.to_conservative(W_tmp_R)

        return UL, UR

    def ppm(self, U: np.ndarray, dx: float, interface_method='simple') -> tuple:
        """Perform Piecewise Parabolic Method (PPM) reconstruction.
                
        # interface_method: Choose between 'simple' and 'full'

        Constructs a parabolic profile in each cell using primitive variables,
        applies monotonicity constraints, and reconstructs interface states.

        Args:
            U (np.ndarray): Conservative variables, shape (n_vars, n_cells).
            dx (float): Grid spacing.

        Returns:
            tuple: Left and right reconstructed states (UL, UR), each shape (n_vars, n_cells - 1).
        """
        n_vars, n_cells = U.shape
        UL = np.zeros((n_vars, n_cells - 1))
        UR = np.zeros((n_vars, n_cells - 1))
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n_cells)]).T
        
        # Store left and right interface values for all cells and variables
        W_L_all = np.zeros((n_vars, n_cells))
        W_R_all = np.zeros((n_vars, n_cells))

        for j in range(n_vars):
            if interface_method == 'simple':
                # Initial interface estimate using centered difference
                for i in range(1, n_cells - 1):
                    delta_W = (W[j, i + 1] - W[j, i - 1]) / 2
                    W_R_all[j, i] = W[j, i] - 0.5 * delta_W
                    W_L_all[j, i] = W[j, i] + 0.5 * delta_W

                # # Extrapolate to boundary cells
                # W_R_all[j, 0] = W[j, 0]
                # W_L_all[j, 0] = W[j, 0]
                # W_R_all[j, n_cells - 1] = W[j, n_cells - 1]
                # W_L_all[j, n_cells - 1] = W[j, n_cells - 1]

                # Higher-order extrapolation at boundaries
                W_R_all[j, 0] = W[j, 1] - 0.5 * (W[j, 2] - W[j, 1])
                W_L_all[j, 0] = W[j, 1] + 0.5 * (W[j, 2] - W[j, 1])
                W_R_all[j, n_cells - 1] = W[j, n_cells - 2] - 0.5 * (W[j, n_cells - 2] - W[j, n_cells - 3])
                W_L_all[j, n_cells - 1] = W[j, n_cells - 2] + 0.5 * (W[j, n_cells - 2] - W[j, n_cells - 3])

            elif interface_method == 'full':
                # Compute interface values for interior cells (5-point stencil)
                for i in range(2, n_cells - 2):
                    W_R_all[j, i] = (7/12) * (W[j, i] + W[j, i + 1]) - (1/12) * (W[j, i - 1] + W[j, i + 2])
                    W_L_all[j, i] = (7/12) * (W[j, i - 1] + W[j, i]) - (1/12) * (W[j, i - 2] + W[j, i + 1])

                # 4-point interpolation at boundaries
                # Left boundary (i=0)
                W_R_all[j, 0] = (1/3) * (W[j, 1] + W[j, 2] + W[j, 3]) - (1/6) * W[j, 4]
                W_L_all[j, 0] = W[j, 0]  # Use ghost cell value
                # Left boundary (i=1)
                W_R_all[j, 1] = (7/12) * (W[j, 1] + W[j, 2]) - (1/12) * (W[j, 0] + W[j, 3])
                W_L_all[j, 1] = (1/3) * (W[j, 0] + W[j, 1] + W[j, 2]) - (1/6) * W[j, 3]
                # Right boundary (i=n_cells-1)
                W_L_all[j, n_cells - 1] = (1/3) * (W[j, n_cells - 2] + W[j, n_cells - 3] + W[j, n_cells - 4]) - (1/6) * W[j, n_cells - 5]
                W_R_all[j, n_cells - 1] = W[j, n_cells - 1]  # Use ghost cell value
                # Right boundary (i=n_cells-2)
                W_L_all[j, n_cells - 2] = (7/12) * (W[j, n_cells - 3] + W[j, n_cells - 2]) - (1/12) * (W[j, n_cells - 4] + W[j, n_cells - 1])
                W_R_all[j, n_cells - 2] = (1/3) * (W[j, n_cells - 3] + W[j, n_cells - 4] + W[j, n_cells - 5]) - (1/6) * W[j, n_cells - 6]

            else:
                raise ValueError("Invalid interface reconstruction method. Choose 'simple' or 'full'.")
        
            # Apply monotonicity constraints
            for i in range(0, n_cells):
                if (W_R_all[j, i] - W[j, i]) * (W[j, i] - W_L_all[j, i]) <= 0:
                    # Extrema Check
                    W_L_all[j, i] = W[j, i]
                    W_R_all[j, i] = W[j, i]
                else:
                    # Overshoot Check
                    delta_W = W_R_all[j, i] - W_L_all[j, i] + 1e-10
                    if (W_R_all[j, i] - W[j, i]) * (W[j, i] - W_L_all[j, i]) > delta_W**2 / 6.0:
                        W_L_all[j, i] = 3 * W[j, i] - 2 * W_R_all[j, i]
                    if (W_R_all[j, i] - W[j, i]) * (W[j, i] - W_L_all[j, i]) < -delta_W**2 / 6.0:
                        W_R_all[j, i] = 3 * W[j, i] - 2 * W_L_all[j, i]

                # # Construct Parabolic Profile (Optional)
                # xi = 0.5  # Evaluate at interface
                # W_6 = 6 * (W[j, i] - 0.5 * (W_L_all[j, i] + W_R_all[j, i]))
                # W_x = W_L_all[j, i] + xi * (W_R_all[j, i] - W_L_all[j, i]) + xi * (1 - xi) * (W_L_all[j, i] - W_R_all[j, i] + W_6)

        # Reconstruct interface states at x_{i+1/2}
        for i in range(0, n_cells - 1):
            W_tmp_L = W_R_all[:, i].copy()  # Left state: right interface of cell i
            W_tmp_R = W_L_all[:, i + 1].copy()  # Right state: left interface of cell i+1
            UL[:, i] = self.equation_system.to_conservative(W_tmp_L)
            UR[:, i] = self.equation_system.to_conservative(W_tmp_R)

        return UL, UR

    def weno5(self, U: np.ndarray, dx: float) -> tuple:
        """Perform WENO5 reconstruction.

        Uses weighted combination of three stencils for high-order accuracy.
        3 ghost cells align with PPM; 2 ghost cells suffice for WENO5 but may limit boundary options.

        Args:
            U (np.ndarray): Conservative variables [n_vars, n_cells].
            dx (float): Spatial grid spacing.

        Returns:
            tuple: Left and right reconstructed states (UL, UR).
        """
        n_vars, n_cells = U.shape
        UL = np.zeros((n_vars, n_cells - 1))
        UR = np.zeros((n_vars, n_cells - 1))
        W = np.array([self.equation_system.to_primitive(U[:, i]) for i in range(n_cells)]).T
        epsilon = max(1e-6, 1e-10 * np.max(np.abs(W)))  # Regularization parameter

        for i in range(2, n_cells - 2):
            for j in range(n_vars):
                if 2 <= i <= n_cells - 3:  # Interior cells: full WENO5
                    v = W[j, i - 2:i + 3]
                    # Smoothness indicators
                    beta0 = 13.0 / 12.0 * (v[0] - 2 * v[1] + v[2])**2 + 0.25 * (v[0] - 4 * v[1] + 3 * v[2])**2
                    beta1 = 13.0 / 12.0 * (v[1] - 2 * v[2] + v[3])**2 + 0.25 * (v[1] - v[3])**2
                    beta2 = 13.0 / 12.0 * (v[2] - 2 * v[3] + v[4])**2 + 0.25 * (3 * v[2] - 4 * v[3] + v[4])**2
                    # Nonlinear weights
                    d0, d1, d2 = 0.1, 0.6, 0.3
                    alpha0 = d0 / (beta0 + epsilon)**2
                    alpha1 = d1 / (beta1 + epsilon)**2
                    alpha2 = d2 / (beta2 + epsilon)**2
                    sum_alpha = alpha0 + alpha1 + alpha2
                    omega0 = alpha0 / sum_alpha
                    omega1 = alpha1 / sum_alpha
                    omega2 = alpha2 / sum_alpha
                    # Candidate polynomials for W_L
                    p0 = (2 * v[0] - 7 * v[1] + 11 * v[2]) / 6.0
                    p1 = (-v[1] + 5 * v[2] + 2 * v[3]) / 6.0
                    p2 = (2 * v[2] + 5 * v[3] - v[4]) / 6.0
                    W_L = omega0 * p0 + omega1 * p1 + omega2 * p2
                    # Candidate polynomials for W_R
                    p0 = (-v[0] + 5 * v[1] + 2 * v[2]) / 6.0
                    p1 = (2 * v[1] + 5 * v[2] - v[3]) / 6.0
                    p2 = (11 * v[2] - 7 * v[3] + 2 * v[4]) / 6.0
                    W_R = omega0 * p0 + omega1 * p1 + omega2 * p2
                else:  # Boundary cells: linear extrapolation
                    if i == 0:  # Left boundary (x_{1/2})
                        W_L = W[j, 1] - 0.5 * (W[j, 2] - W[j, 1])
                        W_R = W[j, 2] + 0.5 * (W[j, 2] - W[j, 1])
                    elif i == 1:  # Near left (x_{3/2})
                        W_L = W[j, 2] - 0.5 * (W[j, 3] - W[j, 2])
                        W_R = W[j, 3] + 0.5 * (W[j, 3] - W[j, 2])
                    elif i == n_cells - 2:  # Right boundary (x_{n_cells-1/2})
                        W_L = W[j, n_cells - 2] + 0.5 * (W[j, n_cells - 2] - W[j, n_cells - 3])
                        W_R = W[j, n_cells - 1] - 0.5 * (W[j, n_cells - 1] - W[j, n_cells - 2])
                    elif i == n_cells - 3:  # Near right (x_{n_cells-3/2})
                        W_L = W[j, n_cells - 3] + 0.5 * (W[j, n_cells - 3] - W[j, n_cells - 4])
                        W_R = W[j, n_cells - 2] - 0.5 * (W[j, n_cells - 2] - W[j, n_cells - 3])

                W_tmp_L = W[:, i].copy()
                W_tmp_R = W[:, i + 1].copy()
                W_tmp_L[j] = W_L
                W_tmp_R[j] = W_R
                UL[:, i] = self.equation_system.to_conservative(W_tmp_L)
                UR[:, i] = self.equation_system.to_conservative(W_tmp_R)

        return UL, UR
