import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

class Metapop:
    '''Metapopulation model

    M -> Patch Number
    N -> Trajectory Number
    C -> Compartments (excluding implied S)


    params:
        mobility_matrix:  NxN
        compartment transitions: C x C. From column (j) to row (i)

    '''
    def __init__(self, start, mobility_matrix, compartment_matrix, infection_func):
        # infer number of trajectories based on parameter dimensions
        self.N = 1
        self.M, self.C = mobility_matrix.shape[1], compartment_matrix.shape[1]
        if len(mobility_matrix.shape) == 3:
            self.N = mobility_matrix.shape[0]
        if type(infection_func) != list:
            self.infect_func = [infection_func for _ in range(self.N)]

        self.R = mobility_matrix.reshape((self.N, self.M, self.M))
        self.T = compartment_matrix.reshape((self.N, self.C, self.C))
        self.rho0 = np.array(start).reshape((self.N, self.M, self.C))

    def run(self, T):
        self.traj = np.zeros((self.N, T, self.M, self.C))
        self.traj[:,0, :, :] = self.rho0
        for t in tqdm(range(0, T - 1)):
            # compute effective pops
            neff = self.traj[:,t].reshape((self.N, self.M, 1, self.C)) * self.R.transpose().reshape((self.N, self.M, self.M, 1))
            # compute infected prob
            self.infect_prob = [self.infect_func[i](neff[i]) for i in range(self.N)]
            # infect them
            self.traj[:, t + 1, :, 0] = (1 - np.sum(self.traj[:,t], axis=-1)) * np.einsum('ijk,ik->ij', self.R, self.infect_prob)
            # move across compartments
            self.traj[:,t + 1] += np.clip(np.einsum('ijk,ikl->ijl', self.traj[:,t], self.T),  0, 1)
        # now add back implied susceptible compartment
        S = 1 - np.sum(self.traj, axis=-1)
        return np.concatenate((S[:,:,:,np.newaxis], self.traj), axis=-1)

def contact_infection_func(beta):
    def fxn(neff):
        p = 1 - np.exp(np.log(1 - beta) * np.sum((neff[:,:,1] + neff[:,:,2]), axis=1))
        return p
    return fxn