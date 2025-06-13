import numpy as np
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Tuple, Union, Optional


def create_grid(
    xmin: float, xmax: float, nx: int, stretch_factor: float = 1.0, dim: int = 1
) -> np.ndarray:
    """
    Generate a non-uniform grid for 1D domains.

    Args:
        xmin: Minimum coordinate.
        xmax: Maximum coordinate.
        nx: Number of grid points.
        stretch_factor: Grid stretching factor (currently linear).
        dim: Dimension of the grid (only 1D supported).

    Returns:
        np.ndarray: Grid coordinates.

    Raises:
        ValueError: If dimension is not 1.
    """
    if dim != 1:
        raise ValueError("Only 1D grids are supported currently.")

    x = np.zeros(nx + 1)

    for i in range(nx + 1):
        # dx = (1 - np.cos(np.pi * i / nx)) / 2
        dx = i / nx
        x[i] = xmin + (xmax - xmin) * dx * stretch_factor

    return x


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
        if config["equation"] == "euler"
        else 1.4
    )
    config["k"] = (
        get_text(root.find("equation/k"), default=1.0, cast=float)
        if config["equation"] == "isentropic"
        else 1.0
    )

    # Boundary conditions
    config["bc_type"] = get_text(
        root.find("boundary_conditions/type"), default="dirichlet"
    )

    # ghsot cells
    config["ghost_cells"] = get_text(
        root.find("solver_settings/ghost_cells"), default=1, cast=int
    )

    # time integration
    config["time_integration"] = get_text(
        root.find("solver_settings/time_integration"), default="euler"
    )

    def parse_values_list(parent: Optional[ET.Element]) -> np.ndarray:
        if parent is not None:
            return np.array([float(v.text) for v in parent if v.text is not None])
        return np.array([])

    config["left_values"] = parse_values_list(
        root.find("boundary_conditions/left_values")
    )
    config["right_values"] = parse_values_list(
        root.find("boundary_conditions/right_values")
    )

    # Initial conditions
    ic_left_elem = root.find("initial_conditions/left")
    ic_right_elem = root.find("initial_conditions/right")
    ic_split_elem = root.find("initial_conditions/split")
    config["initial_conditions"] = {
        "left": parse_values_list(ic_left_elem),
        "right": parse_values_list(ic_right_elem),
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
            while i < len(lines) and not lines[i].startswith("#"):
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
