import unittest
from src.limiters import Limiter


class TestLimiters(unittest.TestCase):
    """Unit tests for Limiter class methods."""

    def test_limiter_minmod(self):
        """Test minmod limiter behavior."""
        limiter = Limiter('minmod')
        self.assertEqual(limiter.limit(1.0, 2.0), 1.0)  # Same sign, smaller magnitude
        self.assertEqual(limiter.limit(-1.0, 2.0), 0.0)  # Opposite signs
        self.assertEqual(limiter.limit(2.0, 1.0), 1.0)  # Same sign, smaller magnitude

    def test_limiter_superbee(self):
        """Test superbee limiter behavior."""
        limiter = Limiter('superbee')
        self.assertEqual(limiter.limit(1.0, 2.0), 2.0)  # Maximizes within constraints
        self.assertEqual(limiter.limit(-1.0, 2.0), 0.0)  # Opposite signs
        self.assertEqual(limiter.limit(0.5, 1.0), 1.0)  # Maximizes within constraints

    def test_limiter_vanleer(self):
        """Test van Leer limiter behavior."""
        limiter = Limiter('vanleer')
        self.assertAlmostEqual(limiter.limit(1.0, 2.0), 4.0 / 3.0, places=5)  # Harmonic mean
        self.assertEqual(limiter.limit(-1.0, 2.0), 0.0)  # Opposite signs
        self.assertAlmostEqual(limiter.limit(1.0, 1.0), 1.0, places=5)  # Equal slopes

    def test_limiter_none(self):
        """Test no limiter behavior."""
        limiter = Limiter('none')
        self.assertEqual(limiter.limit(1.0, 2.0), 1.0)  # Returns first slope
        self.assertEqual(limiter.limit(-1.0, 2.0), -1.0)  # Returns first slope

    def test_limiter_mc(self):
        limiter = Limiter('mc')
        self.assertEqual(limiter.limit(1, 2), 1.5)
        self.assertEqual(limiter.limit(0.5, 1), 0.75)
        self.assertEqual(limiter.limit(1, -1), 0)
        
    def test_limiter_koren(self):
        limiter = Limiter('koren')
        self.assertEqual(limiter.limit(1, 2), 4/3)
        self.assertEqual(limiter.limit(0.5, 1), 2/3)
        self.assertEqual(limiter.limit(1, -1), 0)

    def test_limiter_osher(self):
        limiter = Limiter('osher')
        self.assertEqual(limiter.limit(1, 2), 1)
        self.assertEqual(limiter.limit(2, 1), 2)
        self.assertEqual(limiter.limit(1, -1), 0)

    def test_limiter_sweby(self):
        limiter = Limiter('sweby')
        self.assertEqual(limiter.limit(1, 2), 1.5)
        self.assertEqual(limiter.limit(2, 1), 1.5)
        self.assertEqual(limiter.limit(1, -1), 0)

    def test_limiter_umist(self):
        limiter = Limiter('umist')
        self.assertEqual(limiter.limit(1, 2), 5/4)
        self.assertEqual(limiter.limit(0.5, 1), 5/8)
        self.assertEqual(limiter.limit(1, -1), 0)

if __name__ == '__main__':
    unittest.main()
