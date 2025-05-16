import unittest
from src.limiters import Limiter


class TestLimiters(unittest.TestCase):
    """Unit tests for Limiter class methods."""

    def test_minmod_limiter(self):
        """Test minmod limiter behavior."""
        limiter = Limiter('minmod')
        self.assertEqual(limiter.limit(1.0, 2.0), 1.0)  # Same sign, smaller magnitude
        self.assertEqual(limiter.limit(-1.0, 2.0), 0.0)  # Opposite signs
        self.assertEqual(limiter.limit(2.0, 1.0), 1.0)  # Same sign, smaller magnitude

    def test_superbee_limiter(self):
        """Test superbee limiter behavior."""
        limiter = Limiter('superbee')
        self.assertEqual(limiter.limit(1.0, 2.0), 2.0)  # Maximizes within constraints
        self.assertEqual(limiter.limit(-1.0, 2.0), 0.0)  # Opposite signs
        self.assertEqual(limiter.limit(0.5, 1.0), 1.0)  # Maximizes within constraints

    def test_vanleer_limiter(self):
        """Test van Leer limiter behavior."""
        limiter = Limiter('vanleer')
        self.assertAlmostEqual(limiter.limit(1.0, 2.0), 4.0 / 3.0, places=5)  # Harmonic mean
        self.assertEqual(limiter.limit(-1.0, 2.0), 0.0)  # Opposite signs
        self.assertAlmostEqual(limiter.limit(1.0, 1.0), 1.0, places=5)  # Equal slopes

    def test_none_limiter(self):
        """Test no limiter behavior."""
        limiter = Limiter('none')
        self.assertEqual(limiter.limit(1.0, 2.0), 1.0)  # Returns first slope
        self.assertEqual(limiter.limit(-1.0, 2.0), -1.0)  # Returns first slope

def test_mc(limiter):
    assert limiter.mc(1, 2) == 1.5
    assert limiter.mc(0.5, 1) == 0.75
    assert limiter.mc(1, -1) == 0

def test_koren(limiter):
    assert limiter.koren(1, 2) == 4/3
    assert limiter.koren(0.5, 1) == 2/3
    assert limiter.koren(1, -1) == 0

def test_osher(limiter):
    assert limiter.osher(1, 2) == 1
    assert limiter.osher(2, 1) == 2
    assert limiter.osher(1, -1) == 0

def test_sweby(limiter):
    assert limiter.sweby(1, 2) == 1.5
    assert limiter.sweby(2, 1) == 1.5
    assert limiter.sweby(1, -1) == 0

def test_umist(limiter):
    assert limiter.umist(1, 2) == 5/4
    assert limiter.umist(0.5, 1) == 5/8
    assert limiter.umist(1, -1) == 0

def test_get_limiter(limiter):
    assert limiter.get_limiter('minmod') == limiter.minmod
    with pytest.raises(ValueError):
        limiter.get_limiter('invalid')

if __name__ == '__main__':
    unittest.main()