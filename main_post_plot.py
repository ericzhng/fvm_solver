from src.solution import read_solution, animate_solution


def main():
    """Read and animate solution from file."""
    variable_names, grid, times, solutions = read_solution("solution.dat")
    animate_solution(variable_names, grid, times, solutions, interval=100)


if __name__ == "__main__":
    main()
