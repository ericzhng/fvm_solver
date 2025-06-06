# Hyperbolic Solvers

A Python package for solving hyperbolic conservation laws (shallow water and Euler equations) using Godunov-type schemes.

## Installation

```bash
pip install -r requirements.txt
```

## List of repo

ApproximateRiemannSolvers
CFD_1D_Sod_Shock_Tube
hydro1d
ppmpy
Python-shock-tube


## Usage

Run tests for shallow water or Euler equations:


```bash
python tests/test_shallow_water.py
python tests/test_euler.py
```

Run unit tests:

```bash
pytest tests/
```

## Structure

- `src/`: Core solver code
  - `equation.py`: Equation system definitions
  - `limiters.py`: Slope limiters
  - `reconstructions.py`: State reconstruction methods
  - `fluxes.py`: Numerical flux methods
  - `solver.py`: Main Godunov solver
- `tests/`: Unit and integration tests
  - `test_limiters.py`: Limiter tests
  - `test_reconstructions.py`: Reconstruction tests
  - `test_fluxes.py`: Flux tests
  - `test_shallow_water.py`: Shallow water integration tests
  - `test_euler.py`: Euler integration tests

## Features

- Supports shallow water and Euler equations
- Flux methods: Lax-Friedrichs, Rusanov, FORCE, HLL, HLLC, Roe
- Reconstruction methods: Piecewise constant, MUSCL, PPM, WENO5
- Limiters: Minmod, Superbee, Van Leer, MC, Koren, Osher, Sweby, UMIST
- Boundary conditions: Transmissive, reflective, periodic

