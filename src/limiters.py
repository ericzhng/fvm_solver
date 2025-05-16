import numpy as np


class Limiter:
    """Class to apply slope limiters for MUSCL reconstruction.

    Supports minmod, superbee, van Leer, and no limiting.
    """
    def __init__(self, limiter_type: str = 'minmod'):
        """Initialize the limiter.

        Args:
            limiter_type (str): Type of limiter ('minmod', 'superbee', 'vanleer', 'none').

        Raises:
            ValueError: If limiter_type is not supported.
        """
        self.limiter_type = limiter_type.lower()
        self.limiters = {
            'minmod': self.minmod,
            'superbee': self.superbee,
            'van_leer': self.van_leer,
            'mc': self.mc,
            'koren': self.koren,
            'osher': self.osher,
            'sweby': self.sweby,
            'umist': self.umist
        }

        if self.limiter_type not in self.limiters:
            raise ValueError(f"Unsupported limiter: {self.limiter_type}. Choose from {list(self.limiters.keys())}")

    def minmod(self, a: float, b: float) -> float:
        """Minmod limiter: most dissipative.

        Returns the smallest slope in magnitude if slopes have same sign, else 0.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        if a * b <= 0:
            return 0.0
        return min(abs(a), abs(b)) * np.sign(a)

    def superbee(self, a: float, b: float) -> float:
        """Superbee limiter: least dissipative.

        Maximizes slope within stability constraints.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        if a * b <= 0:
            return 0.0
        return max(min(2 * abs(a), abs(b)), min(abs(a), 2 * abs(b))) * np.sign(a)

    def van_leer(self, a: float, b: float) -> float:
        """Van Leer limiter: smooth intermediate.

        Uses harmonic mean of slopes.

        Args:
            a (float): First slope.
            b (float): Second slope.

        Returns:
            float: Limited slope.
        """
        if a * b <= 0:
            return 0.0
        return 2 * a * b / (a + b + 1e-10)

    def mc(a: float, b: float) -> float:
        """Monotonized Central limiter: Balanced, less diffusive than minmod.
        
        Args:
            a: Left slope
            b: Right slope
            
        Returns:
            Limited slope
        """
        if a * b > 0:
            return max(0, min((a + b) / 2, 2 * a, 2 * b))
        return 0

    def koren(a: float, b: float) -> float:
        """Koren limiter: Third-order in smooth regions.
        
        Args:
            a: Left slope
            b: Right slope
            
        Returns:
            Limited slope
        """
        if a * b > 0:
            return max(0, min(2 * a, (2 * a + b) / 3, 2 * b))
        return 0

    def osher(a: float, b: float, beta: float = 2.0) -> float:
        """Osher limiter: Adjustable sharpness via beta.
        
        Args:
            a: Left slope
            b: Right slope
            beta: Sharpness parameter (1 to 2)
            
        Returns:
            Limited slope
        """
        if a * b > 0:
            return max(0, min(a, beta * b))
        return 0

    def sweby(a: float, b: float, beta: float = 1.5) -> float:
        """Sweby limiter: Tunable between minmod and superbee.
        
        Args:
            a: Left slope
            b: Right slope
            beta: Sharpness parameter (1 to 2)
            
        Returns:
            Limited slope
        """
        if a * b > 0:
            return max(0, min(beta * a, b), min(a, beta * b))
        return 0

    def umist(a: float, b: float) -> float:
        """UMIST limiter: Smooth, similar to Koren.
        
        Args:
            a: Left slope
            b: Right slope
            
        Returns:
            Limited slope
        """
        if a * b > 0:
            return max(0, min(2 * a, (a + 3 * b) / 4, (3 * a + b) / 4, 2 * b))
        return 0

