import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.utils import read_solution


def animate_solution(data, variable_names=[], interval=100):
    # Setup figure and lines
    fig, ax = plt.subplots()
    x = data[0][1]
    W0 = data[0][2]
    # ensure variable_names and dims are matched
    lines = []
    if W0.ndim == 1:
        (line,) = ax.plot(x, W0, label=variable_names[0])
        lines.append(line)
    else:
        if len(variable_names) != W0.shape[0]:
            raise ValueError("variable names counts not equal to W0 dimensions")
        for i, var in enumerate(variable_names):
            (line,) = ax.plot(x, W0[i, :], label=var)
            lines.append(line)
    ax.set_xlabel("x")
    ax.set_ylabel("Variables")
    # Use ax.set_title instead of ax.text for the title
    ax.set_title(f"Solution at t = {data[0][0]:.6f}")
    ax.legend()
    ax.grid(True)

    def update(frame):
        time, _, W = data[frame]
        for i, line in enumerate(lines):
            if W0.ndim == 1:
                line.set_ydata(W)
            else:
                line.set_ydata(W[i, :])
        ax.set_title(f"Solution at t = {time:.6f}")
        return lines

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=interval,
        blit=False,
        repeat=True,
        repeat_delay=3000,
    )
    plt.show()


def main():
    """Read and animate solution from file."""
    data = read_solution("solution.dat")
    # animate_solution(data, ['density'], interval=100)
    animate_solution(data, ["density", "velocity"], interval=100)


if __name__ == "__main__":
    main()
