import pytest
import numpy as np
from src.equation import ShallowWaterSystem
from src.reconstructions import Reconstruction

@pytest.fixture
def shallow_water():
    return ShallowWaterSystem(g=9.81, h_min=1e-10)

@pytest.fixture
def reconstruction(shallow_water):
    return Reconstruction(shallow_water, 'minmod')

def test_piecewise_constant(reconstruction):
    U_ext = np.array([[1, 1, 1, 0.1, 0.1], [0, 0, 0, 0, 0]])
    UL, UR = reconstruction.piecewise_constant(U_ext, dx=0.01)
    assert UL.shape == (2, 4)
    assert UR.shape == (2, 4)
    assert np.allclose(UL[:, 2], [1, 0])
    assert np.allclose(UR[:, 2], [0.1, 0])

def test_muscl(reconstruction):
    U_ext = np.array([[1, 1, 1, 0.1, 0.1], [0, 0, 0, 0, 0]])
    UL, UR = reconstruction.muscl(U_ext, dx=0.01)
    assert UL.shape == (2, 4)
    assert UR.shape == (2, 4)
    assert UL[0, 2] >= 1e-10
    assert UR[0, 2] >= 1e-10

def test_ppm(reconstruction):
    U_ext = np.array([[1, 1, 1, 0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0, 0, 0, 0]])
    UL, UR = reconstruction.ppm(U_ext, dx=0.01)
    assert UL.shape == (2, 6)
    assert UR.shape == (2, 6)
    assert UL[0, 2] >= 1e-10
    assert UR[0, 2] >= 1e-10

def test_weno5(reconstruction):
    U_ext = np.array([[1, 1, 1, 0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0, 0, 0, 0]])
    UL, UR = reconstruction.weno5(U_ext, dx=0.01)
    assert UL.shape == (2, 6)
    assert UR.shape == (2, 6)
    assert UL[0, 2] >= 1e-10
    assert UR[0, 2] >= 1e-10