import numpy as np


class Limiter:
    """Applies slope limiters for MUSCL reconstruction to ensure monotonicity.

    Supports minmod, superbee, van Leer, MC, Koren, Osher, Sweby, UMIST, and no limiting.
    Limiters prevent oscillations near discontinuities in finite volume schemes.

    Attributes:
        limiter_type (str): Type of limiter ('minmod', 'superbee', 'vanleer', 'mc', 'koren', 'osher', 'sweby', 'umist', 'none').
        beta (float): Sharpness parameter for Osher and Sweby limiters (1 to 2).
    """

    def __init__(self, limiter_type: str = 'minmod', beta: float = 1.5):
        """Initialize the limiter.

        Args:
            limiter_type (str): Type of limiter ('minmod', 'superbee', 'vanleer', 'mc', 'koren', 'osher', 'sweby', 'umist', 'none').
            beta (float, optional): Sharpness parameter for Osher/Sweby (default: 1.5).

        Raises:
            ValueError: If limiter_type is unsupported or beta is out of range [1, 2].
        """
        self.limiter_type = limiter_type.lower()
        self.beta = max(1.0, min(2.0, beta))  # Ensure beta in [1, 2]
        self.limiters = {
            'minmod': self.minmod,
            'superbee': self.superbee,
            'vanleer': self.van_leer,
            'mc': self.mc,
            'koren': self.koren,
            'osher': lambda a, b: self.osher(a, b, self.beta),
            'sweby': lambda a, b: self.sweby(a, b, self.beta),
            'umist': self.umist,
            'none': self.no_limiter
        }
        if self.limiter_type not in self.limiters:
            raise ValueError(f"Unsupported limiter: {self.limiter_type}. Choose from {list(self.limiters.keys())}")

    def limit(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply the limiter to two slopes.

        Args:
            a (np.ndarray): First slope (left or forward difference).
            b (np.ndarray): Second slope (right or backward difference).

        Returns:
            np.ndarray: Limited slope, same shape as inputs.
        """
        # Vectorized application of limiter
        return np.vectorize(self.limiters[self.limiter_type], otypes=[float])(a, b)

    def minmod(self, a: float, b: float) -> float:
        """Minmod limiter: most dissipative, returns smallest slope if same sign.

        Formula: min(|a|, |b|) * sign(a) if a*b > 0, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        return np.sign(a) * min(abs(a), abs(b)) if a * b > 0 else 0.0

    def superbee(self, a: float, b: float) -> float:
        """Superbee limiter: least dissipative, maximizes slope within stability.

        Formula: max(min(2|a|, |b|), min(|a|, 2|b|)) * sign(a) if a*b > 0, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        return np.sign(a) * max(min(2 * abs(a), abs(b)), min(abs(a), 2 * abs(b))) if a * b > 0 else 0.0

    def van_leer(self, a: float, b: float) -> float:
        """Van Leer limiter: smooth, uses harmonic mean of slopes.

        Formula: 2ab/(a+b) if a*b > 0, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        return 2 * a * b / (a + b + 1e-10) if a * b > 0 else 0.0

    def mc(self, a: float, b: float) -> float:
        """Monotonized Central limiter: balanced, less diffusive than minmod.

        Formula: max(0, min((a+b)/2, 2a, 2b)) if a*b > 0, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        return max(0, min((a + b) / 2, 2 * a, 2 * b)) if a * b > 0 else 0.0

    def koren(self, a: float, b: float) -> float:
        """Koren limiter: third-order in smooth regions.

        Formula: max(0, min(2a, (2a+b)/3, 2b)) if a*b > 0, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        return max(0, min(2 * a, (2 * a + b) / 3, 2 * b)) if a * b > 0 else 0.0

    def osher(self, a: float, b: float, beta: float) -> float:
        """Osher limiter: adjustable sharpness via beta.

        Formula: max(0, min(a, beta*b)) if a*b > 0, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.
            beta (float): Sharpness parameter (1 to 2).

        Returns:
            float: Limited slope.
        """
        return max(0, min(a, beta * b)) if a * b > 0 else 0.0

    def sweby(self, a: float, b: float, beta: float) -> float:
        """Sweby limiter: tunable between minmod and superbee.

        Formula: max(0, min(beta*a, b), min(a, beta*b)) if a*b > 0, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.
            beta (float): Sharpness parameter (1 to 2).

        Returns:
            float: Limited slope.
        """
        return max(0, min(beta * a, b), min(a, beta * b)) if a * b > 0 else 0.0

    def umist(self, a: float, b: float) -> float:
        """UMIST limiter: smooth, similar to Koren.

        Formula: max(0, min(2a, (a+3b)/4, (3a+b)/4, 2b)) if a*b > 0, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        return max(0, min(2 * a, (a + 3 * b) / 4, (3 * a + b) / 4, 2 * b)) if a * b > 0 else 0.0

    def no_limiter(self, a: float, b: float) -> float:
        """No limiter: returns first slope unchanged.

        Args:
            a (float): First slope.
            b (float): Second slope (ignored).

        Returns:
            float: Unchanged first slope.
        """
        return a
