import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def dirichlet_mat_layer(input, start, name):
    '''Dirichlet distributed trainable distribution (columns sum to 1). Zeros in starting matrix are preserved'''
    x  = TrainableInputLayer(start, constraint=PositiveMaskedConstraint(start > 0), name=name + '-hypers')(input)
    x = tf.keras.layers.Lambda(lambda x: x +  1e-10 * np.mean(start), name=name + '-jitter')(x)
    return tfp.layers.DistributionLambda(lambda t: tfd.Dirichlet(10 * t[0]), name=name + '-dist')(x)

def normal_mat_layer(input, start, name, start_var=1, clip_high=100):
    '''Normally distributed trainable distribution. Zeros in starting matrix are preserved'''
    # stack variance
    start_val = np.concatenate((
            start[np.newaxis,...],
            np.tile(start_var, start.shape)[np.newaxis,...]
            ))
    # zero-out variance of zeroed starts
    start_val[1] = start_val[1] * (start_val[0] > 0)
    x  = TrainableInputLayer(start_val, name=name + '-hypers', constraint=PositiveMaskedConstraint(start_val > 0))(input)
    x = tf.keras.layers.Lambda(lambda x: x + 1e-10 * np.mean(start), name=name + '-jitter')(x)
    return tfp.layers.DistributionLambda(
            lambda t: tfd.TruncatedNormal(
                loc=t[0,0],
                low=0.0,
                high=clip_high,
                scale=t[0,1]),
                name=name + '-dist'
        )(x)


class TrainableMetaModel(tf.keras.Model):
    def __init__(self, start, mobility_matrix, compartment_matrix, infect_func, timesteps, loss_fxn):
        super(TrainableMetaModel, self).__init__()
        self.R_layer = TrainableInputLayer(mobility_matrix,
                                       constraint=NormalizationConstraint(1, mobility_matrix > 0))
        self.T_layer = TrainableInputLayer(compartment_matrix, NormalizationConstraint(1, compartment_matrix > 0))
        self.start_layer = TrainableInputLayer(start, tf.keras.constraints.MinMaxNorm(min_value=0.0, rate=0.2, max_value=0.3, axis=-1))
        self.metapop_layer = MetapopLayer(timesteps, infect_func)
        self.traj_layer = AddSusceptibleLayer(name='traj')
        self.agreement_layer = tf.keras.layers.Lambda(loss_fxn)
    def call(self, inputs):
        self.R = self.R_layer(inputs)
        self.T = self.T_layer(inputs)
        self.rho = self.start_layer(inputs)
        x =  self.metapop_layer([self.R, self.T, self.rho])
        self.traj = self.traj_layer(x)
        self.agreement = self.agreement_layer(self.traj)
        return self.traj, self.agreement
    def get_traj(self):
        return self([0.0])[0]
    def compile(self, optimizer, **kwargs):
        if 'loss' in kwargs:
            raise ValueError('Do not specificy loss, instead use loss_fxn in constructor')
        super(TrainableMetaModel, self).compile(optimizer, [None, 'mean_absolute_error'], **kwargs)
    def fit(self, steps=100, **kwargs):
        super(TrainableMetaModel, self).fit(steps * [0.], steps * [0.], batch_size=1, **kwargs)

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

class TrainableInputLayer(tf.keras.layers.Layer):
    ''' Create trainable input layer'''
    def __init__(self, initial_value, constraint=None,regularizer=None, **kwargs):
        super(TrainableInputLayer, self).__init__(**kwargs)
        flat = initial_value.flatten()
        self.w = self.add_weight(
            'value',
            shape=initial_value.shape,
            initializer=tf.constant_initializer(flat),
            constraint=constraint,
            dtype=self.dtype,
            trainable=True)
        self.trainable_flat = len(flat)
    def call(self, inputs):
        return tf.expand_dims(self.w, 0)

class DeltaRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, value, strength=1e-3):
        self.strength = strength
        self.value = value

    def __call__(self, x):
        return self.strength * tf.reduce_sum((x - self.value)**2)


class NormalizationConstraint(tf.keras.constraints.Constraint):
  ''' Makes weights normalized after reshape and applying mask'''

  def __init__(self, axis, mask):
    self.axis = axis
    self.mask = mask

  def __call__(self, w):
    wz = tf.clip_by_value(w, 0., 1e10) * self.mask
    m = tf.reduce_sum(wz, axis=self.axis, keepdims=True)
    return wz / m

class PositiveMaskedConstraint(tf.keras.constraints.Constraint):
  ''' Makes weights normalized after reshape and applying mask'''

  def __init__(self, mask):
    self.mask = mask

  def __call__(self, w):
    wz = tf.math.multiply_no_nan(tf.clip_by_value(w, 0., 1e10), self.mask)
    return wz

class AddSusceptibleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddSusceptibleLayer, self).__init__(**kwargs)
    def call(self, trajs):
        S = 1 - tf.reduce_sum(trajs, axis=-1)
        result = tf.concat((S[:,:,:,tf.newaxis], trajs), axis=-1)
        # want batch index first
        result = tf.transpose(result, perm=[1,0,2,3])
        return result

class MetapopLayer(tf.keras.layers.Layer):
    def __init__(self, timesteps, infect_func, dtype=tf.float32):
        super(MetapopLayer, self).__init__(dtype=dtype)
        self.infect_func = infect_func
        self.timesteps = timesteps
    def build(self, input_shape):
        self.N, self.M, self.C = input_shape[-1]

    def call(self, inputs):
        R, T, rho0 = inputs
        trajs_array = tf.TensorArray(size=self.timesteps, element_shape=(self.N, self.M, self.C), dtype=self.dtype)
        def body(i, prev_rho, trajs_array):
            # compute effective pops
            neff = tf.reshape(prev_rho, (-1, self.M, 1, self.C)) *\
                   tf.reshape(tf.transpose(R), (-1, self.M, self.M, 1))
            ntot = tf.reduce_sum(R, axis=1)
            # compute infected prob
            infect_prob = self.infect_func(neff, ntot)
            # infect them
            new_infected = (1 - tf.reduce_sum(prev_rho, axis=-1)) * tf.einsum('ijk,ik->ij', R, infect_prob)
            # create new compartment values
            rho = tf.einsum('ijk,ikl->ijl', prev_rho, T) + \
                new_infected[:,:,tf.newaxis] * tf.constant([1] + \
                [0 for _ in range(self.C - 1)], dtype=self.dtype)
            # project back to allowable values
            rho = tf.clip_by_value(rho, 0, 1e10)
            #rho /= tf.clip_by_value(tf.reduce_sum(rho, axis=-1, keepdims=True), 1, 1000000)
            # write
            trajs_array = trajs_array.write(i, rho)
            return i + 1, rho, trajs_array
        cond = lambda i, *_: i < self.timesteps
        _, _, trajs_array = tf.while_loop(cond, body, (0, rho0, trajs_array))
        return trajs_array.stack()

def contact_infection_func(beta, infectious_compartments):
    if type(beta) == float:
        beta = tf.constant([beta])
    def fxn(neff, ntot):
        ninf = tf.zeros_like(neff[:, :, 0])
        for i in infectious_compartments:
            ninf += neff[:, :, i]
        p = 1 - tf.math.exp(tf.math.log(1 - beta[:,tf.newaxis]) * tf.reduce_sum((ninf) / ntot[:,:,tf.newaxis], axis=2))
        return p
    return fxn

