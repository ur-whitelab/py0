import numpy as np

class SIRModel:
    def __init__(self, initial_compartments, alpha, beta, N_steps=1000):
        self.I, self.R = initial_compartments[1], initial_compartments[2]
        self.alpha, self.beta = alpha, beta
        self.trajectory = []
        self.N_steps = N_steps

    def _step(self):
        I_next = min(1., max(0., self.I + self.beta * self.I *
                             (1.-self.I-self.R) - self.alpha * self.I))
        R_next = min(1., max(0., self.R + self.alpha * self.I))
        self.I, self.R = I_next, R_next

    def _run(self):
        self.trajectory.append([1. - self.I - self.R, self.I, self.R])
        for _ in range(self.N_steps-1):
            self._step()
            self.trajectory.append([1. - self.I - self.R, self.I, self.R])

    def _get_trajectory(self):
        return np.array(self.trajectory)

    def __call__(self):
        self._run()
        return(self._get_trajectory())
