import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_solution(filename='solution.dat'):
    """Read solution from ASCII file and return time steps and data.

    Args:
        filename (str): Path to solution file.

    Returns:
        list: List of (time, x, W) tuples for each time step.
    """
    data = []
    current_step = None
    current_time = None
    x = []
    W = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Step'):
                if current_step is not None:
                    data.append((current_time, np.array(x), np.array(W).T))
                    x, W = [], []
                _, step, _, time = line.split()
                current_step = int(step)
                current_time = float(time)
            elif line.startswith('# x'):
                continue
            elif line and not line.startswith('#'):
                values = [float(v) for v in line.split()]
                x.append(values[0])
                W.append(values[1:])
    
    if current_step is not None:
        data.append((current_time, np.array(x), np.array(W).T))
    
    return data

def animate_solution(data, variable_names=['density', 'velocity', 'pressure'], interval=100):
    """Animate solution over time steps.

    Args:
        data (list): List of (time, x, W) tuples.
        variable_names (list): Names of variables to plot.
        interval (int): Delay between frames in milliseconds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = data[0][1]
    lines = [ax.plot(x, data[0][2][i, :], label=name)[0] for i, name in enumerate(variable_names)]
    
    ax.set_xlabel('x')
    ax.set_ylabel('Variables')
    ax.set_title(f'Solution at t = {data[0][0]:.6f}')
    ax.legend()
    ax.grid(True)
    
    def update(frame):
        time, x, W = data[frame]
        for i, line in enumerate(lines):
            line.set_ydata(W[i, :])
        ax.set_title(f'Solution at t = {time:.6f}')
        return lines
    
    ani = animation.FuncAnimation(fig, update, frames=len(data), interval=interval, blit=True)
    plt.show()

def main():
    """Read and animate solution from file."""
    data = read_solution('solution.dat')
    animate_solution(data)

if __name__ == '__main__':
    main()