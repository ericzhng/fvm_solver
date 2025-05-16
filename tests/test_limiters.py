import pytest
import numpy as np
from src.limiters import Limiter

@pytest.fixture
def limiter():
    return Limiter()

def test_minmod(limiter):
    assert limiter.minmod(1, 2) == 1
    assert limiter.minmod(-1, -2) == -1
    assert limiter.minmod(1, -1) == 0

def test_superbee(limiter):
    assert limiter.superbee(1, 2) == 2
    assert limiter.superbee(0.5, 1) == 1
    assert limiter.superbee(1, -1) == 0

def test_van_leer(limiter):
    assert abs(limiter.van_leer(1, 1) - 1) < 1e-10
    assert limiter.van_leer(1, -1) == 0
    assert abs(limiter.van_leer(2, 4) - 8/3) < 1e-10

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