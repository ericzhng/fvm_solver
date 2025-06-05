
import numpy as np
import xml.etree.ElementTree as ET

def gen_grid(xmin, xmax, nx, stretch_factor=1.0, dim=1) -> np.ndarray:
    """Generate a non-uniform grid for 1D, 2D, or 3D domains.

    Args:
        xmin (float): Minimum coordinate.
        xmax (float): Maximum coordinate.
        nx (int): Number of grid points.
        stretch_factor (float): Grid stretching factor.
        dim (int): Dimension of the grid (1, 2, or 3).

    Returns:
        np.ndarray: Grid coordinates.
    """
    x = np.zeros(nx + 1)
    for i in range(nx + 1):
        # dx = (1 - np.cos(np.pi * i / nx)) / 2
        dx = i / nx
        x[i] = xmin + (xmax - xmin) * dx * stretch_factor
    if dim == 1:
        return x
    else:
        raise ValueError("Dimension must be 1, 2, or 3")

def get_text(element, default=None, required=False, cast=None):
    if element is not None and element.text is not None:
        try:
            return cast(element.text) if cast else element.text
        except Exception:
            if required:
                raise ValueError(f"Invalid value for element {element.tag}")
            return default
    if required:
        raise ValueError(f"Missing required element or text for {element.tag if element is not None else 'unknown'}")
    return default

def parse_xml_config(filename):
    """Parse simulation configuration from XML file.

    Args:
        filename (str): Path to XML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    config = {}

    config['dimension'] = get_text(root.find('geometry/dimension'), default=1, required=True, cast=int)
    config['xmin'] = get_text(root.find('geometry/domain/xmin'), default=0.0, required=True, cast=float)
    config['xmax'] = get_text(root.find('geometry/domain/xmax'), default=1.0, required=True, cast=float)
    config['nx'] = get_text(root.find('mesh/nx'), default=100, required=True, cast=int)
    config['stretch_factor'] = get_text(root.find('mesh/stretch_factor'), default=1.0, cast=float)
    config['equation'] = get_text(root.find('equation/type'), default='euler')
    config['gamma'] = get_text(root.find('equation/gamma'), default=1.4, cast=float) if config['equation'] == 'euler' else 1.4
    config['k'] = get_text(root.find('equation/k'), default=1.0, cast=float) if config['equation'] == 'isentropic' else 1.0
    config['bc_type'] = get_text(root.find('boundary_conditions/type'), default='dirichlet')

    left_values_elem = root.find('boundary_conditions/left_values')
    config['left_values'] = np.array([float(v.text) for v in left_values_elem if v.text is not None] if left_values_elem is not None else [])

    right_values_elem = root.find('boundary_conditions/right_values')
    config['right_values'] = np.array([float(v.text) for v in right_values_elem if v.text is not None] if right_values_elem is not None else [])

    ic_euler_left_elem = root.find('initial_conditions/euler/left')
    ic_euler_right_elem = root.find('initial_conditions/euler/right')
    ic_euler_split_elem = root.find('initial_conditions/euler/split')
    ic_isentropic_left_elem = root.find('initial_conditions/isentropic/left')
    ic_isentropic_right_elem = root.find('initial_conditions/isentropic/right')
    ic_isentropic_split_elem = root.find('initial_conditions/isentropic/split')
    ic_swe_left_elem = root.find('initial_conditions/shallow_water/left')
    ic_swe_right_elem = root.find('initial_conditions/shallow_water/right')
    ic_swe_split_elem = root.find('initial_conditions/shallow_water/split')
    ic_adv_left_elem = root.find('initial_conditions/advection/left')
    ic_adv_right_elem = root.find('initial_conditions/advection/right')
    ic_adv_split_elem = root.find('initial_conditions/advection/split')
    config['initial_conditions'] = {
        'euler': {
            'left': np.array([float(v.text) for v in ic_euler_left_elem if v.text is not None] if ic_euler_left_elem is not None else []),
            'right': np.array([float(v.text) for v in ic_euler_right_elem if v.text is not None] if ic_euler_right_elem is not None else []),
            'split': float(ic_euler_split_elem.text) if ic_euler_split_elem is not None and ic_euler_split_elem.text is not None else 0.5
        },
        'isentropic': {
            'left': np.array([float(v.text) for v in ic_isentropic_left_elem if v.text is not None] if ic_isentropic_left_elem is not None else []),
            'right': np.array([float(v.text) for v in ic_isentropic_right_elem if v.text is not None] if ic_isentropic_right_elem is not None else []),
            'split': float(ic_isentropic_split_elem.text) if ic_isentropic_split_elem is not None and ic_isentropic_split_elem.text is not None else 0.5
        },
        'shallow_water': {
            'left': np.array([float(v.text) for v in ic_swe_left_elem if v.text is not None] if ic_swe_left_elem is not None else []),
            'right': np.array([float(v.text) for v in ic_swe_right_elem if v.text is not None] if ic_swe_right_elem is not None else []),
            'split': float(ic_swe_split_elem.text) if ic_swe_split_elem is not None and ic_swe_split_elem.text is not None else 0.5
        },
        'advection': {
            'left': np.array([float(v.text) for v in ic_adv_left_elem if v.text is not None] if ic_adv_left_elem is not None else []),
            'right': np.array([float(v.text) for v in ic_adv_right_elem if v.text is not None] if ic_adv_right_elem is not None else []),
            'split': float(ic_adv_split_elem.text) if ic_adv_split_elem is not None and ic_adv_split_elem.text is not None else 0.5
        }
    }
    config['T'] = get_text(root.find('solver_settings/T'), default=1.0, cast=float)
    config['cfl'] = get_text(root.find('solver_settings/cfl'), default=0.5, cast=float)
    config['flux'] = get_text(root.find('solver_settings/flux'), default='roe')
    config['reconstruction'] = get_text(root.find('solver_settings/reconstruction'), default='linear')
    config['reconstruction_vars'] = get_text(root.find('solver_settings/reconstruction_vars'), default='primitive')
    config['limiter'] = get_text(root.find('solver_settings/limiter'), default='minmod')
    config['max_iterations'] = get_text(root.find('solver_settings/max_iterations'), default=10000, cast=int)
    config['convergence_tolerance'] = get_text(root.find('solver_settings/convergence_tolerance'), default=1e-6, cast=float)
    config['output_format'] = get_text(root.find('output/format'), default='csv')
    config['output_filename'] = get_text(root.find('output/filename'), default='output.csv')

    return config

def read_solution(filename):
    # Reads the solution.dat file and returns a list of (time, x, W) tuples
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('# Step'):
            # Parse time
            time_line = line
            time = float(time_line.split('Time')[1].strip())
            # Skip header
            i += 2
            x_list, w_list = [], []
            while i < len(lines) and not lines[i].startswith('#'):
                vals = lines[i].split()
                if len(vals) == 4:
                    x_list.append(float(vals[0]))
                    w_list.append([float(vals[1]), float(vals[2]), float(vals[3])])
                elif len(vals) == 3:
                    x_list.append(float(vals[0]))
                    w_list.append([float(vals[1]), float(vals[2])])
                elif len(vals) == 2:
                    x_list.append(float(vals[0]))
                    w_list.append(float(vals[1]))
                i += 1
            x = np.array(x_list)
            W = np.array(w_list).T  # shape: (3, N)
            data.append((time, x, W))
        else:
            i += 1
    return data
