import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
                i += 1
            x = np.array(x_list)
            W = np.array(w_list).T  # shape: (3, N)
            data.append((time, x, W))
        else:
            i += 1
    return data

def animate_solution(data, variable_names=['density', 'velocity', 'pressure'], interval=100):
    # Setup figure and lines
    fig, ax = plt.subplots()
    x = data[0][1]
    W0 = data[0][2]
    lines = []
    for i, var in enumerate(variable_names):
        (line,) = ax.plot(x, W0[i, :], label=var)
        lines.append(line)
    ax.set_xlabel('x')
    ax.set_ylabel('Variables')
    # Use ax.set_title instead of ax.text for the title
    ax.set_title(f'Solution at t = {data[0][0]:.6f}')
    ax.legend()
    ax.grid(True)

    def update(frame):
        time, _, W = data[frame]
        for i, line in enumerate(lines):
            line.set_ydata(W[i, :])
        ax.set_title(f'Solution at t = {time:.6f}')
        return lines

    ani = animation.FuncAnimation(
        fig, update, frames=len(data), interval=interval, blit=False, repeat=True, repeat_delay=3000
    )
    plt.show()

def main():
    """Read and animate solution from file."""
    data = read_solution('solution.dat')
    animate_solution(data, ['density'], interval=100)

if __name__ == '__main__':
    main()
