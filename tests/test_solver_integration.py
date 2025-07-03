"""
This module contains integration tests for the FVM solver.

It is designed to run a matrix of tests, combining different physics equations
(from config files) with various numerical methods (fluxes, reconstructions)
to ensure the stability and correctness of the solver framework.
"""

import os
import sys
import pytest
import numpy as np

# Add the project root directory to the Python path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import parse_xml_config, create_grid
from src.solver import Solver
from src.boundary import BoundaryCondition
from src.reconstruction import Reconstruction
from src.equation.equation_advection import EqnAdvection
from src.equation.equation_burgers import EqnBurgers
from src.equation.equation_euler import EqnEuler
from src.equation.equation_isentropic_gas import EqnIsentropicGas
from src.equation.equation_keyfitz_kranzer import EqnKK
from src.equation.equation_shallow_water import EqnShallowWater
from src.equation.equation_traffic_flow import EqnTrafficLWR

# --- Test Configuration ---

# List of configuration files to test, representing different physics problems
CONFIG_FILES = [
    "config/advection_step_response.xml",
    "config/burgers_step.xml",
    "config/euler_sod_shock_tube.xml",
    "config/isentropic_density_jump.xml",
    "config/keyfitz_kranzer_step.xml",
    "config/shallow_water_dam_break.xml",
    "config/traffic_flow_step.xml",
]

# List of numerical methods to test for each problem
FLUX_METHODS = ["roe", "hllc", "ausm", "hlle"]
RECONSTRUCTION_METHODS = ["constant", "muscl"]


# --- Test Fixtures ---


@pytest.fixture(scope="module", params=CONFIG_FILES)
def config_path(request):
    """Pytest fixture to provide the path to each configuration file."""
    return request.param


@pytest.fixture(scope="module")
def config(config_path):
    """Pytest fixture to parse the XML config file and provide the data."""
    return parse_xml_config(config_path)


# --- Test Functions ---


@pytest.mark.parametrize("flux_method", FLUX_METHODS)
@pytest.mark.parametrize("reconstruction_method", RECONSTRUCTION_METHODS)
def test_solver_integration(config, flux_method, reconstruction_method):
    """
    Runs a full solver simulation for a given configuration and numerical method.

    This test checks if the solver completes without runtime errors. It does not
    verify the correctness of the solution, but serves as a stability and
    integration check.

    Args:
        config (dict): The parsed configuration dictionary.
        flux_method (str): The numerical flux scheme to test.
        reconstruction_method (str): The spatial reconstruction method to test.
    """
    print(
        f"Testing with config: {config["equation"]}, flux: {flux_method}, reconstruction: {reconstruction_method}"
    )

    # --- Setup from Config ---
    equation_dict = {
        "advection": EqnAdvection(speed=config.get("speed", 1.0)),
        "burgers": EqnBurgers(),
        "euler": EqnEuler(gamma=config.get("gamma", 1.4)),
        "isentropic": EqnIsentropicGas(
            gamma=config.get("gamma", 1.4), k=config.get("k", 1.0)
        ),
        "keyfitz_kranzer": EqnKK(),
        "shallow_water": EqnShallowWater(gravity=config.get("g", 9.81)),
        "traffic_flow": EqnTrafficLWR(
            rho_max=config.get("rho_max", 1.0), v_max=config.get("v_max", 1.0)
        ),
    }
    equation = equation_dict[config["equation"].lower()]

    mesh_grid = create_grid(
        config["xmin"], config["xmax"], config["nx"], config.get("stretch_factor", 1.0)
    )

    bc_inst = BoundaryCondition(
        eqn_obj=equation,
        bc_kind=config["bc"]["type"],
        grid=mesh_grid,
        n_ghost=config["ghost_cells"],
        left_boundary_state=config["bc"]["left"],
        right_boundary_state=config["bc"]["right"],
    )

    reconst_inst = Reconstruction(
        eqn_obj=equation,
        str_reconst=reconstruction_method,
        str_flux=flux_method,
        str_limiter=config.get("limiter", "minmod"),
        str_domain=config.get("reconstruction_vars", "primitive"),
    )

    n_vars = equation.num_vars
    W0 = np.zeros((n_vars, config["nx"]), dtype=float)
    split_idx = np.argmin(np.abs(mesh_grid - config["ic"]["split"]))
    W0[:, :split_idx] = np.array(config["ic"]["left"])[:, np.newaxis]
    W0[:, split_idx:] = np.array(config["ic"]["right"])[:, np.newaxis]
    U0 = equation.to_conservative_batch(W0)

    solver_inst = Solver(
        eqn_obj=equation,
        bc_obj=bc_inst,
        mesh_obj=mesh_grid,
        reconst_obj=reconst_inst,
        n_ghost=config["ghost_cells"],
        cfl=config["cfl"],
        max_iterations=5,  # Keep iterations low for testing
        convergence_tol=1e-4,
        output_filename=f"test_{config["equation"]}_{flux_method}_{reconstruction_method}.dat",
    )

    # --- Run Test ---
    try:
        _, _ = solver_inst.solve(U0, config["T"], "euler")
        assert True
    except Exception as e:
        pytest.fail(
            f"Solver failed for {config["equation"]} with {flux_method}/{reconstruction_method}. Error: {e}"
        )
