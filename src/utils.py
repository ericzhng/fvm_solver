"""
This module provides utility functions for the FVM solver, including grid
generation, XML configuration parsing, solution data handling, and plotting.
"""

import numpy as np
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def create_grid(
    xmin: float, xmax: float, nx: int, grading: float = 1.0, edge_grading: bool = True
) -> np.ndarray:
    """
    Generates a 1D grid with optional non-uniform grading.

    This function can create a uniform grid or a grid stretched towards one
    end or the center, similar to OpenFOAM's blockMesh utility.

    Args:
        xmin (float): The minimum coordinate of the grid.
        xmax (float): The maximum coordinate of the grid.
        nx (int): The number of cells in the grid.
        grading (float, optional): The ratio of the last cell width to the first.
                                 > 1: Cells get larger towards xmax.
                                 < 1: Cells get smaller towards xmax.
                                 Defaults to 1.0 (uniform).
        edge_grading (bool, optional): If True, grading is applied from one edge
                                     to the other. If False, grading is applied
                                     from the edges towards the center.
                                     Defaults to True.

    Returns:
        np.ndarray: An array of grid node coordinates of shape (nx + 1,).
    """
    L = xmax - xmin
    if grading == 1.0 or nx == 1:
        return np.linspace(xmin, xmax, nx + 1)

    if edge_grading:
        # Standard one-sided grading
        if grading < 1.0:
            # Invert grading to handle refinement towards xmin consistently
            r = 1.0 / grading
            reverse = True
        else:
            r = grading
            reverse = False

        # The ratio between consecutive cells is constant, r_cell = r^(1/(nx-1))
        r_cell = r ** (1.0 / (nx - 1))
        # Sum of a geometric series to find the first cell width
        denom = (1.0 - r_cell**nx) / (1.0 - r_cell) if r_cell != 1.0 else float(nx)
        dx0 = L / denom

        points = [0.0]
        for i in range(nx):
            points.append(points[-1] + dx0 * (r_cell**i))
        mesh = np.array(points)

        if reverse:
            mesh = L - mesh[::-1]
    else:
        # Symmetric grading towards the center
        raise NotImplementedError("Center grading is not yet fully implemented.")

    # Scale and shift to the specified domain
    mesh = xmin + (mesh / mesh[-1]) * L
    return mesh


def plot_1d_mesh(mesh_nodes: np.ndarray):
    """
    Visualizes the 1D mesh nodes as vertical lines.

    Args:
        mesh_nodes (np.ndarray): An array of mesh node coordinates.
    """
    x_min, x_max = np.min(mesh_nodes), np.max(mesh_nodes)
    margin = 0.05 * (x_max - x_min)

    plt.figure(figsize=(10, 2))
    for x in mesh_nodes:
        plt.axvline(x, color="blue", linestyle="-", linewidth=1.0)

    plt.xlim(x_min - margin, x_max + margin)
    plt.ylim(-1, 1)
    plt.yticks([])
    plt.xlabel("x-coordinate")
    plt.title("1D Mesh Visualization")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.show()


def get_text(
    element: Optional[ET.Element],
    default: Any = None,
    required: bool = False,
    cast: Optional[Callable[[str], Any]] = None,
) -> Any:
    """
    Safely extracts and casts text content from an XML element.

    Args:
        element (Optional[ET.Element]): The parent XML element to search within.
        tag (str): The tag of the child element to find.
        default (Any, optional): The default value to return if the element is
                                 not found or has no text. Defaults to None.
        required (bool, optional): If True, a ValueError is raised if the element
                                   is missing or empty. Defaults to False.
        cast_to (Optional[Callable], optional): A function (e.g., int, float) to
                                               cast the text content to.
                                               Defaults to None.

    Returns:
        Any: The extracted and casted value, or the default.

    Raises:
        ValueError: If a required element is missing or its value is invalid.
    """
    if element is not None and element.text is not None:
        try:
            return cast(element.text) if cast else element.text
        except Exception:
            if required:
                raise ValueError(f"Invalid value for element {element.tag}")
            return default
    if required:
        raise ValueError(
            f"Missing required element or text for {element.tag if element is not None else 'unknown'}"
        )
    return default


def parse_xml_values(element: Optional[ET.Element]) -> np.ndarray:
    """
    Parses a list of floating-point values from child elements.

    Args:
        element (Optional[ET.Element]): The parent XML element.
        tag (str): The tag of the child elements containing the values.

    Returns:
        np.ndarray: An array of the parsed float values.
    """
    if element is not None:
        return np.array([float(v.text) for v in element if v.text is not None])
    return np.array([])


def parse_xml_config(filename: str) -> Dict[str, Any]:
    """
    Parses a simulation configuration from a specified XML file.

    Args:
        filename (str): The path to the XML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed configuration parameters.
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    config: Dict[str, Any] = {}

    # Geometry and mesh
    config["dimension"] = get_text(
        root.find("geometry/dimension"), default=1, required=True, cast=int
    )
    config["xmin"] = get_text(
        root.find("geometry/domain/xmin"), default=0.0, required=True, cast=float
    )
    config["xmax"] = get_text(
        root.find("geometry/domain/xmax"), default=1.0, required=True, cast=float
    )
    config["nx"] = get_text(root.find("mesh/nx"), default=100, required=True, cast=int)
    config["stretch_factor"] = get_text(
        root.find("mesh/stretch_factor"), default=1.0, cast=float
    )

    # Equation
    config["equation"] = get_text(root.find("equation/type"), default="euler")
    config["gamma"] = (
        get_text(root.find("equation/gamma"), default=1.4, cast=float)
        if config["equation"] == "euler" or config["equation"] == "isentropic"
        else 1.4
    )
    config["k"] = (
        get_text(root.find("equation/k"), default=1.0, cast=float)
        if config["equation"] == "isentropic"
        else 1.0
    )
    config["g"] = (
        get_text(root.find("equation/g"), default=9.82, cast=float)
        if config["equation"] == "shallow_water"
        else 9.82
    )
    config["speed"] = (
        get_text(root.find("equation/speed"), default=1.0, cast=float)
        if config["equation"] == "advection"
        else 1.0
    )
    config["rhoMax"] = (
        get_text(root.find("equation/rhoMax"), default=1.0, cast=float)
        if config["equation"] == "traffic_flow"
        else 1.0
    )
    config["vMax"] = (
        get_text(root.find("equation/vMax"), default=10.0, cast=float)
        if config["equation"] == "traffic_flow"
        else 10.0
    )

    # Boundary conditions
    config["bc"] = {
        "type": get_text(root.find("boundary_conditions/type"), default="dirichlet"),
        "left": parse_xml_values(root.find("boundary_conditions/left_values")),
        "right": parse_xml_values(root.find("boundary_conditions/right_values")),
    }

    # ghsot cells
    config["ghost_cells"] = get_text(
        root.find("solver_settings/ghost_cells"), default=1, cast=int
    )

    # time integration
    config["time_integration"] = get_text(
        root.find("solver_settings/time_integration"), default="euler"
    )

    # Initial conditions
    ic_split_elem = root.find("initial_conditions/split")
    config["ic"] = {
        "left": parse_xml_values(root.find("initial_conditions/left")),
        "right": parse_xml_values(root.find("initial_conditions/right")),
        "split": (
            float(ic_split_elem.text)
            if ic_split_elem is not None and ic_split_elem.text is not None
            else 0.5
        ),
    }

    # Solver settings
    config["T"] = get_text(root.find("solver_settings/T"), default=1.0, cast=float)
    config["cfl"] = get_text(root.find("solver_settings/cfl"), default=0.5, cast=float)
    config["flux"] = get_text(root.find("solver_settings/flux"), default="roe")
    config["reconstruction"] = get_text(
        root.find("solver_settings/reconstruction"), default="linear"
    )
    config["reconstruction_vars"] = get_text(
        root.find("solver_settings/reconstruction_vars"), default="primitive"
    )
    config["limiter"] = get_text(root.find("solver_settings/limiter"), default="minmod")
    config["max_iterations"] = get_text(
        root.find("solver_settings/max_iterations"), default=10000, cast=int
    )
    config["convergence_tolerance"] = get_text(
        root.find("solver_settings/convergence_tolerance"), default=1e-6, cast=float
    )

    # Output
    config["output_format"] = get_text(root.find("output/format"), default="csv")
    config["output_filename"] = get_text(
        root.find("output/filename"), default="output.csv"
    )

    return config


def read_solution(filename: str) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Reads and parses a solution data file generated by the solver.

    The file is expected to contain data blocks for each time step, with each
    block starting with a header line like '# Step 1, Time 0.0123'.

    Args:
        filename (str): The path to the solution data file.

    Returns:
        List[Tuple[float, np.ndarray, np.ndarray]]: A list of tuples, where
            each tuple contains (time, x_coordinates, W_solution).
            W_solution has a shape of (num_vars, num_cells).
    """
    all_data: List[Tuple[float, np.ndarray, np.ndarray]] = []
    try:
        with open(filename, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Solution file not found at {filename}")
        return []

    # Split the file content into blocks for each time step
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        # Find the header line to extract time
        header_line = next((line for line in lines if line.startswith("# x")), None)
        if not header_line:
            # Fallback for older format
            header_line = next(
                (line for line in lines if line.startswith("# Step")), None
            )

        # This format is deprecated but supported for backward compatibility
        if header_line and "Time" in header_line:
            try:
                time = float(header_line.split("Time")[1].strip())
            except (IndexError, ValueError):
                continue  # Skip malformed blocks
        else:
            # In the new format, time is not in the header, so we can use a placeholder
            # or decide on another way to handle it. For now, let's use block index.
            time = float(len(all_data))  # Placeholder time

        # Load numerical data, skipping header lines
        data_lines = [line for line in lines if not line.startswith("#")]
        if not data_lines:
            continue

        try:
            data_matrix = np.loadtxt(data_lines)
            if data_matrix.ndim == 1:
                data_matrix = data_matrix.reshape(1, -1)

            x_coords = data_matrix[:, 0]
            solution = data_matrix[:, 1:].T
            all_data.append((time, x_coords, solution))
        except ValueError:
            # Skip blocks that cannot be parsed as numerical data
            continue

    return all_data


def numerical_jacobian(
    F: Callable[[np.ndarray], np.ndarray], U: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    """
    Computes the numerical Jacobian of a vector function using central differences.

    The Jacobian matrix J of a function F at a point U is given by:
    J_ij = ∂F_i / ∂U_j

    Args:
        F (Callable[[np.ndarray], np.ndarray]): The vector function for which to
                                                 compute the Jacobian.
        U (np.ndarray): The state vector at which to evaluate the Jacobian.
        h (float, optional): The perturbation size for the finite difference.
                             Defaults to 1e-6.

    Returns:
        np.ndarray: The numerical Jacobian matrix.
    """
    n = len(U)
    jacobian = np.zeros((n, n))
    I = np.identity(n)

    for j in range(n):
        # Perturb U in the j-th direction
        U_plus = U + h * I[:, j]
        U_minus = U - h * I[:, j]

        # Compute the j-th column of the Jacobian using central difference
        jacobian[:, j] = (F(U_plus) - F(U_minus)) / (2 * h)

    return jacobian
