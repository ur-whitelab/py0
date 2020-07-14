import numpy as np
import tensorflow as tf
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

class MetaModel:
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
        # in case arrays are passed
        start, mobility_matrix, compartment_matrix = np.array(start), np.array(mobility_matrix), np.array(compartment_matrix)
        self.M, self.C = mobility_matrix.shape[1], compartment_matrix.shape[1]
        if len(mobility_matrix.shape) == 3:
            self.N = mobility_matrix.shape[0]
        self.infect_func = infection_func
        self.dtype = tf.float32
        self.R = tf.constant(mobility_matrix.reshape((self.N, self.M, self.M)), dtype=self.dtype)
        self.T = tf.constant(compartment_matrix.reshape((self.N, self.C, self.C)), dtype=self.dtype)
        self.rho0 = tf.constant(np.array(start).reshape((self.N, self.M, self.C)), dtype=self.dtype)

    def run(self, time, display_tqdm=True):
        trajs_array = tf.TensorArray(size=time, element_shape=self.rho0.shape, dtype=self.dtype)
        def body(i, prev_rho, trajs_array):
            # compute effective pops
            neff = tf.reshape(prev_rho, (self.N, self.M, 1, self.C)) *\
                   tf.reshape(tf.transpose(self.R), (self.N, self.M, self.M, 1))
            ntot = tf.reduce_sum(self.R, axis=1)
            # compute infected prob
            infect_prob = self.infect_func(neff, ntot)
            # infect them
            new_infected = (1 - tf.reduce_sum(prev_rho, axis=-1)) * tf.einsum('ijk,ik->ij', self.R, infect_prob)
            # create new compartment values
            rho = tf.einsum('ijk,ikl->ijl', prev_rho, self.T) + \
                new_infected[:,:,tf.newaxis] * tf.constant([1] + [0 for _ in range(self.C - 1)], dtype=self.dtype)
            # project back to allowable values
            rho = tf.clip_by_value(rho, 0, 100000)
            #rho /= tf.clip_by_value(tf.reduce_sum(rho, axis=-1, keepdims=True), 1, 1000000)
            # write
            trajs_array = trajs_array.write(i, rho)
            return i + 1, rho, trajs_array
        cond = lambda i, *_: i < time
        _, rho, trajs_array = tf.while_loop(cond, body, (0, self.rho0, trajs_array))
        trajs = trajs_array.stack()
        trajs_array.close()
        # now add back implied susceptible compartment
        S = 1 - tf.reduce_sum(trajs, axis=-1)
        result = tf.concat((S[:,:,:,tf.newaxis], trajs), axis=-1)
        # want batch index first
        with tf.device('/CPU:0'):
            result = tf.transpose(result, perm=[1,0,2,3])
        return result

def contact_infection_func(beta):
    def fxn(neff, ntot):
        p = 1 - tf.math.exp(tf.math.log(1 - beta[:,tf.newaxis]) * tf.reduce_sum((neff[:,:,:,1] + neff[:,:,:,2]) / ntot[:,:,tf.newaxis], axis=2))
        return p
    return fxn