"""
This module defines the Limiter class, which provides a collection of slope
limiters for use in MUSCL-type finite volume schemes to ensure stability and
prevent spurious oscillations near discontinuities.
"""

import numpy as np


class Limiter:
    """
    A factory for various slope limiter functions used in high-resolution
    finite volume methods.

    Slope limiters are essential for MUSCL-type schemes to enforce the Total
    Variation Diminishing (TVD) property, preventing the formation of new
    extrema in the solution and suppressing numerical oscillations (Gibbs'
    phenomenon) that can occur near shocks or sharp gradients.

    This class provides a unified interface to select and apply a limiter.

    Attributes:
        name (str): The name of the selected limiter.
        beta (float): A sharpness parameter used by certain limiters like
                      Osher and Sweby, controlling their dissipative nature.
        limiters (dict): A dictionary mapping limiter names to their
                         corresponding implementation methods.
    """

    def __init__(self, str_limiter: str = "minmod", beta: float = 1.5):
        """
        Initializes the Limiter object.

        Args:
            str_limiter (str, optional): The name of the limiter to use.
                Supported options include 'minmod', 'superbee', 'vanleer',
                'mc', 'koren', 'osher', 'sweby', 'umist', and 'none'.
                Defaults to "minmod".
            beta (float, optional): The sharpness parameter for limiters that
                support it (e.g., Osher, Sweby). Must be in the range [1, 2].
                Defaults to 1.5.

        Raises:
            ValueError: If an unsupported `str_limiter` is provided.
        """
        self.name = str_limiter.lower()
        # Clamp beta to the valid range [1.0, 2.0]
        self.beta = max(1.0, min(2.0, beta))

        self.limiters = {
            "minmod": self.minmod,
            "superbee": self.superbee,
            "vanleer": self.van_leer,
            "mc": self.mc,
            "koren": self.koren,
            "osher": lambda a, b: self.osher(a, b, self.beta),
            "sweby": lambda a, b: self.sweby(a, b, self.beta),
            "umist": self.umist,
            "none": self.no_limiter,
        }

        if self.name not in self.limiters:
            valid_limiters = list(self.limiters.keys())
            raise ValueError(
                f"Unsupported limiter: '{self.name}'. " f"Choose from {valid_limiters}"
            )

    def limiter_func(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Applies the selected limiter function to a ratio of successive gradients.

        Args:
            a (np.ndarray): First slope (e.g., left/forward difference).
            b (np.ndarray): Second slope (e.g., right/backward difference).

        Returns:
            np.ndarray: The limited slope correction factor, φ(r).
        """
        # Vectorized application of limiter
        if self.name == "mc":
            ret = np.vectorize(self.limiters[self.name], otypes=[float])(a, b, c)
        else:
            ret = np.vectorize(self.limiters[self.name], otypes=[float])(a, b)
        return ret

    def minmod(self, a: float, b: float) -> float:
        """
        The Minmod limiter. It is the most dissipative of the common limiters.
        φ(r) = max(0, min(1, r))
        """
        return np.sign(a) * min(abs(a), abs(b)) if a * b > 0 else 0.0

    def superbee(self, a: float, b: float) -> float:
        """
        The Superbee limiter. It is one of the least dissipative limiters.
        φ(r) = max(0, min(2r, 1), min(r, 2))
        """
        return (
            np.sign(a) * max(min(2 * abs(a), abs(b)), min(abs(a), 2 * abs(b)))
            if a * b > 0
            else 0.0
        )

    def van_leer(self, a: float, b: float) -> float:
        """
        The Van Leer limiter, a smooth choice.
        φ(r) = (r + |r|) / (1 + |r|)
        """
        return 2 * a * b / (a + b + 1e-10) if a * b > 0 else 0.0

    def mc(self, a: float, b: float, c: float) -> float:
        """
        The Monotonized Central (MC) limiter.
        φ(r) = max(0, min((1+r)/2, 2, 2r))
        """
        return max(0, min((a + b) / 2, 2 * a, 2 * b)) if a * b > 0 else 0.0

    def koren(self, a: float, b: float) -> float:
        """
        The Koren limiter, which is third-order accurate in smooth regions.
        φ(r) = max(0, min(2r, (1+2r)/3, 2))
        """
        return max(0, min(2 * a, (2 * a + b) / 3, 2 * b)) if a * b > 0 else 0.0

    def osher(self, a: float, b: float, beta: float) -> float:
        """
        The Osher limiter, with an adjustable sharpness parameter β.
        φ(r) = max(0, min(r, β))
        """
        return max(0, min(a, beta * b)) if a * b > 0 else 0.0

    def sweby(self, a: float, b: float, beta: float) -> float:
        """
        The Sweby limiter, tunable between other limiters via parameter β.
        φ(r) = max(0, min(βr, 1), min(r, β))
        """
        return max(0, min(beta * a, b), min(a, beta * b)) if a * b > 0 else 0.0

    def umist(self, a: float, b: float) -> float:
        """
        The UMIST limiter, a smooth and symmetric choice.
        φ(r) = max(0, min(2r, (1+3r)/4, (3+r)/4, 2))
        """
        return (
            max(0, min(2 * a, (a + 3 * b) / 4, (3 * a + b) / 4, 2 * b))
            if a * b > 0
            else 0.0
        )

    def no_limiter(self, a: float, b: float) -> float:
        """
        A pass-through function that applies no limiting. Equivalent to a
        higher-order, non-TVD scheme.
        φ(r) = r
        """
        return a
