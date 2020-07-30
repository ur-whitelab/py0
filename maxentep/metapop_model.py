import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow_probability.python.internal import tensorshape_util
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

EPS = np.finfo(np.float32).tiny

# NOTE in all these trainable input -> distribution lambdas, we ignore batch dim
# this is because the trainable input layers are not a function of input.

def _compute_trans_diagonal(tri_values, indices, shape):
    # force tri_values to be batched
    m = tf.scatter_nd(indices, tri_values, shape)
    diag = 1 - tf.reduce_sum(m, axis=-1)
    return diag

def recip_norm_mat_dist(trans_times, trans_times_var):
    # would like to do this check, but cannot
    # since it makes this a dynamic layer
    # if tf.rank(trans_times) != 2:
    #    raise ValueError('Input must have shape (N, N)')
    L = tf.shape(trans_times)[-1]
    # This gets a mask ( > 0 ) that is batched
    indices = tf.cast(tf.where(trans_times > 0), tf.int32)
    diag_indices = tf.tile(tf.range(L)[:, tf.newaxis], [1, 2])
    values = tf.gather_nd(trans_times, indices)
    v_vars =  tf.gather_nd(trans_times_var, indices)
    j = tfd.JointDistributionSequentialAutoBatched([
        tfd.Independent(tfd.TransformedDistribution(
            tfd.TruncatedNormal(loc=values, scale=v_vars, low=1., high=1e10),
            bijector=tfb.Reciprocal()
        ),1),
        lambda v: tfd.Independent(
            tfd.Deterministic(loc=_compute_trans_diagonal(v, indices, trans_times.shape))
            ,1),
        lambda d, v: tfd.Independent(
            tfd.Deterministic(
                loc=tf.scatter_nd(
                    tf.concat((indices, diag_indices), axis=0),
                    tf.concat((v, d), axis=0),
                    trans_times.shape))
            ,1),

    ])
    return j

def recip_norm_mat_layer(input, time_means, time_vars, name):
    '''Column Normalized Reciprical Gaussian trainable distribution. Zeros in starting matrix are preserved'''
    # add extra row that we use for concentration
    combined = np.stack((time_means, time_vars))
    x  = TrainableInputLayer(combined, constraint=PositiveMaskedConstraint(combined > 0), name=name + '-hypers')(input)
    return tfp.layers.DistributionLambda(
        lambda t: recip_norm_mat_dist(t[0,0], t[0,1]),
        name=name + '-dist')(x)

def categorical_normal_layer(input, start_logits,  start_mean, start_scale, name):
    x = TrainableInputLayer(start_logits, name=name + '-start-logit-hypers')(input)
    y = tf.keras.layers.Dense(2,
        kernel_initializer=tf.constant_initializer(value=[start_mean, start_scale]),
        name=name + '-norm-logit-hypers',
        dtype=x.dtype)(input)
    return tfp.layers.DistributionLambda(lambda t:
        tfd.JointDistributionSequential([
            tfd.Multinomial(1, t[0]),
            tfd.Normal(loc=t[1][...,0], scale=t[1][...,1]),
            lambda b, n: tfd.Independent(tfd.Deterministic(loc=b * n), 1)
            ])
        )([x,y])

def dirichlet_mat_layer(input, start, name):
    '''Dirichlet distributed trainable distribution (columns sum to 1). Zeros in starting matrix are preserved'''
    # add extra row that we use for concentration
    start_aug = np.concatenate((start, np.zeros_like(start)))
    start_aug[-1, -1] = 5.0
    x  = TrainableInputLayer(start_aug, constraint=PositiveMaskedConstraint(start_aug > 0), name=name + '-hypers')(input)
    x = tf.keras.layers.Lambda(lambda x: x +  1e-10 * np.mean(start), name=name + '-jitter')(x)
    return tfp.layers.DistributionLambda(lambda t: tfd.Dirichlet((t[0,-1,-1] * t[0,:-1])), name=name + '-dist')(x)

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

class ParameterHypers:
    def __init__(self):
        self.beta_low = 0.01
        self.beta_high = 0.7
        self.beta_var = 0.1
        self.start_high = 0.5
        self.start_var = 0.1
        self.R_var = 0.2

class ParameterJoint(tf.keras.Model):
    def __init__(self, start, mobility_matrix,
                 compartment_matrix, beta,
                 name='', hypers=None):
    '''Create trainable joint model for parameters'''
    if hypers is None:
        hypers = ParameterHypers()
    i = tf.keras.layers.Input((1,))
    # infection parameter first
    beta_layer = tf.keras.layers.Dense(
        1,
        use_bias=False,
        kernel_constraint=tf.keras.constraints.MinMaxNorm(hypers.beta_low, hypers.beta_high),
        kernel_initializer = tf.keras.initializers.Constant(beta),
        name='beta')
    beta_dist = tfp.layers.DistributionLambda(
        lambda b: tfd.TruncatedNormal(
            loc=b,
            scale=hypers.beta_var,
            low=0.0,
            high=hypers.beta_high + hypers.beta_var),
        name='beta-dist'
    )(beta_layer(i))
    R_dist = normal_mat_layer(i, mobility_matrix,  start_var=hypers.R_var, name='R-dist')
    T_dist =  dirichlet_mat_layer(i, compartment_matrix, name='T-dist')
    start_dist = normal_mat_layer(i, start, start_var=hypers.start_var, clip_high=hypers.start_high, name='rho-dist')
    super(ParameterJoint, self).__init__((inputs=i, outputs=[R_dist, T_dist, start_dist, beta_dist], name=name + '-model'))

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

class MetaModel(tf.keras.Model):
    def __init__(self, infect_func, timesteps):
        super(MetaModel, self).__init__()
        self.metapop_layer = MetapopLayer(timesteps, infect_func)
        self.traj_layer = AddSusceptibleLayer(name='traj')
    def call(self, R, T, rho, params):
        if tf.rank(R) == 2:
            R, T, rho = R[tf.newaxis,...], T[tf.newaxis, ...], rho[tf.newaxis,...]
        x =  self.metapop_layer([R, T, rho, params])
        traj = self.traj_layer(x)
        return traj

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
        batch_dim = tf.shape(inputs)[:1]
        return tf.tile(self.w[tf.newaxis,...], tf.concat((batch_dim, tf.ones(tf.rank(self.w), dtype=tf.int32)), axis=0))

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
    wz = tf.math.multiply_no_nan(tf.clip_by_value(w, EPS, 1e10), self.mask)
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
        self.N, self.M, self.C = input_shape[2]

    def call(self, inputs):
        R, T, rho0 = inputs[:3]
        infect_params = inputs[3:]
        trajs_array = tf.TensorArray(size=self.timesteps, element_shape=(self.N, self.M, self.C), dtype=self.dtype)
        def body(i, prev_rho, trajs_array):
            # compute effective pops
            neff = tf.reshape(prev_rho, (-1, self.M, 1, self.C)) *\
                   tf.reshape(tf.transpose(R), (-1, self.M, self.M, 1))
            ntot = tf.reduce_sum(R, axis=1)
            # compute infected prob
            infect_prob = self.infect_func(neff, ntot, *infect_params)
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

class ContactInfectionLayer(tf.keras.layers.Layer):
    def __init__(self, initial_beta,  infectious_compartments, clip_low=0.01, clip_high=0.5, **kwargs):
        super(ContactInfectionLayer, self).__init__(**kwargs)
        self.infectious_compartments = infectious_compartments
        self.beta = tf.Variable(
            initial_value=tf.reshape(initial_beta, (-1,)),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, clip_low, clip_high),
            name='contact-beta'
        )
    def call(self, neff, ntot):
        ninf = tf.zeros_like(neff[:, :, 0])
        for i in self.infectious_compartments:
            ninf += neff[:, :, i]
        p = 1 - tf.math.exp(tf.math.log(1 -self.beta[:,tf.newaxis]) * tf.reduce_sum((ninf) / ntot[:,:,tf.newaxis], axis=2))
        return p

def contact_infection_func(infectious_compartments):
    def fxn(neff, ntot, beta):
        ninf = tf.zeros_like(neff[:, :, 0])
        for i in infectious_compartments:
            ninf += neff[:, :, i]
        p = 1 - tf.math.exp(tf.math.log(1 - tf.reshape(beta, (-1,1))) * tf.reduce_sum((ninf) / ntot[:,:,tf.newaxis], axis=2))
        return p
    return fxn


def negloglik(y, rv_y):
    logp = rv_y.log_prob(y + EPS)
    logp = tf.reduce_sum(tf.reshape(logp, (tf.shape(y)[0], -1)), axis=1)
    return -logp