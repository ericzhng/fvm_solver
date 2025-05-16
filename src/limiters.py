import numpy as np

class Limiter:
    """Provides limiter functions for TVD schemes in hyperbolic solvers."""
    
    @staticmethod
    def minmod(a: float, b: float) -> float:
        """Minmod limiter: Most diffusive, ensures TVD property.
        
        Args:
            a: Left slope
            b: Right slope
            
        Returns:
            Limited slope
        """
        if a * b > 0:
            return np.sign(a) * min(abs(a), abs(b))
        return 0

    @staticmethod
    def superbee(a: float, b: float) -> float:
        """Superbee limiter: Sharpest, may introduce oscillations.
        
        Args:
            a: Left slope
            b: Right slope
            
        Returns:
            Limited slope
        """
        if a * b > 0:
            return np.sign(a) * max(min(2 * abs(a), abs(b)), min(abs(a), 2 * abs(b)))
        return 0

    @staticmethod
    def van_leer(a: float, b: float) -> float:
        """Van Leer limiter: Smooth, balances diffusion and sharpness.
        
        Args:
            a: Left slope
            b: Right slope
            
        Returns:
            Limited slope
        """
        if a * b > 0:
            return (a * b + abs(a * b)) / (a + b + 1e-10)
        return 0

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    def get_limiter(self, name: str):
        """Returns the limiter function by name.
        
        Args:
            name: Limiter name (e.g., 'minmod', 'superbee')
            
        Returns:
            Limiter function
            
        Raises:
            ValueError: If limiter name is unsupported
        """
        limiters = {
            'minmod': self.minmod,
            'superbee': self.superbee,
            'van_leer': self.van_leer,
            'mc': self.mc,
            'koren': self.koren,
            'osher': self.osher,
            'sweby': self.sweby,
            'umist': self.umist
        }
        if name not in limiters:
            raise ValueError(f"Unsupported limiter: {name}")
        return limiters[name]