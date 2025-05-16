import pytest
import numpy as np
from src.equation import ShallowWaterSystem
from src.fluxes import Flux

@pytest.fixture
def shallow_water():
    return ShallowWaterSystem(g=9.81, h_min=1e-10)

@pytest.fixture
def flux(shallow_water):
    return Flux(shallow_water)

def test_lax_friedrichs(flux, shallow_water):
    UL = np.array([1, 0])
    UR = np.array([0.1, 0])
    F = flux.lax_friedrichs(UL, UR)
    assert F.shape == (2,)
    assert F[0] >= 0

def test_rusanov(flux, shallow_water):
    UL = np.array([1, 0])
    UR = np.array([0.1, 0])
    F = flux.rusanov(UL, UR)
    assert F.shape == (2,)
    assert F[0] >= 0

def test_force(flux, shallow_water):
    UL = np.array([1, 0])
    UR = np.array([0.1, 0])
    F = flux.force(UL, UR)
    assert F.shape == (2,)
    assert F[0] >= 0

def test_hll(flux, shallow_water):
    UL = np.array([1, 0])
    UR = np.array([0.1, 0])
    F = flux.hll(UL, UR)
    assert F.shape == (2,)
    assert F[0] >= 0

def test_hllc(flux, shallow_water):
    UL = np.array([1, 0])
    UR = np.array([0.1, 0])
    F = flux.hllc(UL, UR)
    assert F.shape == (2,)
    assert F[0] >= 0

def test_roe(flux, shallow_water):
    UL = np.array([1, 0])
    UR = np.array([0.1, 0])
    F = flux.roe(UL, UR)
    assert F.shape == (2,)
    assert F[0] >= 0

def test_get_flux(flux):
    assert flux.get_flux('HLLC') == flux.hllc
    with pytest.raises(ValueError):
        flux.get_flux('invalid')
