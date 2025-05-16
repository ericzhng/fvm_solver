import numpy as np

class EquationSystem:
    def to_primitive(self, U):
        raise NotImplementedError

    def to_conservative(self, W):
        raise NotImplementedError

    # balances computational efficiency and clarity in the numerical implementation
    def compute_flux(self, U, W):
        raise NotImplementedError

    def sound_speed(self, W):
        raise NotImplementedError

    def velocity(self, W):
        raise NotImplementedError

    def get_variable_names(self):
        raise NotImplementedError

class ShallowWaterSystem(EquationSystem):
    def __init__(self, g=9.81, h_min=1e-10):
        self.g = g
        self.h_min = h_min

    def to_primitive(self, U):
        h = max(U[0], self.h_min)
        u = U[1] / h if h > self.h_min else 0.0
        return np.array([h, u])

    def to_conservative(self, W):
        h = max(W[0], self.h_min)
        hu = h * W[1]
        return np.array([h, hu])

    def compute_flux(self, U, W):
        h = max(W[0], self.h_min)
        u = W[1]
        hu = U[1]
        return np.array([hu, hu * u + 0.5 * self.g * h**2])

    def sound_speed(self, W):
        h = max(W[0], self.h_min)
        return np.sqrt(self.g * h)

    def velocity(self, W):
        return np.sqrt(W[1])

    def get_variable_names(self):
        return ['Height', 'Velocity']

class EulerEquationSystem(EquationSystem):
    def __init__(self, gamma=1.4, rho_min=1e-10, p_min=1e-10):
        self.gamma = gamma
        self.rho_min = rho_min
        self.p_min = p_min

    def to_primitive(self, U):
        rho = max(U[0], self.rho_min)
        u = U[1] / rho
        p = max((self.gamma - 1) * (U[2] - 0.5 * rho * u**2), self.p_min)
        return np.array([rho, u, p])

    def to_conservative(self, W):
        rho = max(W[0], self.rho_min)
        p = max(W[2], self.p_min)
        mom = rho * W[1]
        energy = p / (self.gamma - 1) + 0.5 * rho * W[1]**2
        return np.array([rho, mom, energy])

    def compute_flux(self, U, W):
        rho = max(W[0], self.rho_min)
        p = max(W[2], self.p_min)
        u = W[1]
        mom = U[1]
        energy = U[2]
        return np.array([mom, rho * u**2 + p, u * (energy + p)])

    def sound_speed(self, W):
        rho = max(W[0], self.rho_min)
        p = max(W[2], self.p_min)
        return np.sqrt(self.gamma * p / rho)

    def velocity(self, W):
        return np.sqrt(W[1])

    def get_variable_names(self):
        return ['Density', 'Velocity', 'Pressure']
