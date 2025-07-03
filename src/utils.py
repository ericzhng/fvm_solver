import numpy as np
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt

import numpy as np


def create_grid(
    xmin: float, xmax: float, nx: int, grading: float = 1.0, edge_grading: bool = True
) -> np.ndarray:
    """
    Generate a 1D grid mimicking OpenFOAM blockMesh with grading control.

    Args:
        xmin: Minimum coordinate.
        xmax: Maximum coordinate.
        nx: Number of cells.
        grading: Cell size ratio. >1: finer at max, <1: finer at min, =1: uniform.
        edge_grading: True for grading towards edges, False for center grading.
    Returns:
        np.ndarray: Grid node coordinates (nx+1,)
    """
    L = xmax - xmin
    if grading == 1.0 or nx == 1:
        return np.linspace(xmin, xmax, nx + 1)

    if edge_grading:
        # Grading towards edges (OpenFOAM-like)
        if grading < 1.0:
            grading = 1.0 / grading  # Invert for consistency
            reverse = True
        else:
            reverse = False

        # Geometric progression sum for cell sizes
        r = grading ** (1.0 / (nx - 1))
        denom = (1.0 - r**nx) / (1.0 - r) if r != 1.0 else float(nx)
        dx0 = L / denom

        points = [0.0]
        for i in range(nx):
            points.append(points[-1] + dx0 * (r**i))

        mesh = np.array(points)
        if reverse:
            mesh = L - mesh[::-1]
    else:
        # Grading towards center
        n_half = nx // 2
        r = grading if grading >= 1.0 else 1.0 / grading
        dx0 = (L / 2) / sum(r**i for i in range(n_half))

        left = [0.0]
        for i in range(n_half):
            left.append(left[-1] + dx0 * (r**i))

        if nx % 2 == 0:
            right = left[::-1]
            mesh = np.array(left[:-1] + right)
        else:
            right = [left[-1]]
            for i in range(n_half):
                right.append(right[-1] + dx0 * (r ** (n_half - 1 - i)))
            mesh = np.array(left + right[1:])

    # Scale and shift
    mesh = xmin + (mesh / mesh[-1]) * L
    return mesh


def plot_1d_mesh(mesh_nodes):
    """
    Plot the 1D mesh nodes as vertical lines.
    Args:
        mesh_nodes: np.ndarray of mesh node coordinates
    """
    x_min, x_max = np.min(mesh_nodes), np.max(mesh_nodes)
    x_margin = 0.02 * (x_max - x_min)

    for x in mesh_nodes:
        plt.axvline(x, color="b", linestyle="-", linewidth=0.8)

    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("x")
    plt.title("1D Mesh")
    plt.show()


def get_text(
    element: Optional[ET.Element],
    default: Any = None,
    required: bool = False,
    cast: Optional[Callable[[str], Any]] = None,
) -> Any:
    """
    Extract text from an XML element, with optional casting and default value.

    Args:
        element: XML element.
        default: Default value if text is missing.
        required: Whether the text is required.
        cast: Function to cast the text (e.g., int, float).

    Returns:
        The extracted and casted value, or default.

    Raises:
        ValueError: If required text is missing or invalid.
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


def parse_values_list(parent: Optional[ET.Element]) -> np.ndarray:
    if parent is not None:
        return np.array([float(v.text) for v in parent if v.text is not None])
    return np.array([])


def parse_xml_config(filename: str) -> Dict[str, Any]:
    """
    Parse simulation configuration from XML file.

    Args:
        filename: Path to XML configuration file.

    Returns:
        dict: Configuration parameters.
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
        "left": parse_values_list(root.find("boundary_conditions/left_values")),
        "right": parse_values_list(root.find("boundary_conditions/right_values")),
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
        "left": parse_values_list(root.find("initial_conditions/left")),
        "right": parse_values_list(root.find("initial_conditions/right")),
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
    Reads the solution.dat file and returns a list of (time, x, W) tuples.

    Args:
        filename: Path to the solution file.

    Returns:
        List of tuples: (time, x, W), where W shape is (variables, N).
    """
    data: List[Tuple[float, np.ndarray, np.ndarray]] = []
    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# Step"):
            # Parse time
            try:
                time = float(line.split("Time")[1].strip())
            except Exception:
                i += 1
                continue
            i += 2  # Skip header
            x_list, w_list = [], []
            if not lines[i].strip():
                i += 1
            while i < len(lines) and not lines[i].startswith("#") and lines[i].strip():
                vals = [float(v) for v in lines[i].split()]
                x_list.append(vals[0])
                w_list.append(vals[1:] if len(vals) > 2 else vals[1])
                i += 1
            x = np.array(x_list)
            W = (
                np.array(w_list).T
                if w_list and isinstance(w_list[0], list)
                else np.array(w_list)
            )
            data.append((time, x, W))
        else:
            i += 1
    return data


def numerical_jacobian(
    F: Callable[[np.ndarray], np.ndarray], U: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    """
    Compute the numerical Jacobian matrix of F at U using central differences.

    Args:
        F: Function mapping U to F(U).
        U: State vector.
        h: Perturbation size.

    Returns:
        np.ndarray: Jacobian matrix.
    """
    n = len(U)
    A = np.zeros((n, n))
    for j in range(n):
        U_plus = U.copy()
        U_plus[j] += h
        U_minus = U.copy()
        U_minus[j] -= h
        A[:, j] = (F(U_plus) - F(U_minus)) / (2 * h)
    return A
