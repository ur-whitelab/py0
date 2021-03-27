from tensorflow_probability.python.internal import tensorshape_util
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

EPS = np.finfo(np.float32).tiny

# NOTE in all these trainable input -> distribution lambdas, we ignore batch dim
# this is because the trainable input layers are not a function of input.


def _compute_trans_diagonal(tri_values, indices, shape):
    # force tri_values to be batched
    m = tf.scatter_nd(indices, tri_values, shape)
    diag = 1 - tf.reduce_sum(m, axis=-1)
    return diag


def recip_norm_mat_dist(trans_times, trans_times_var, indices, sample_R=True, n_infectious_compartments=1):
    values = tf.gather_nd(trans_times, indices)
    v_vars = tf.gather_nd(trans_times_var, indices)
    if sample_R:
        j = tfd.Independent(
            tfd.TruncatedNormal(loc=tf.clip_by_value(
                values, 1 + 1e-3, 1e10), scale=v_vars, low=0, high=1e10))
    else:
        j = tfd.Independent(tfd.TransformedDistribution(
            tfd.TruncatedNormal(loc=tf.clip_by_value(
                values, 1 + 1e-3, 1e10), scale=v_vars, low=n_infectious_compartments, high=1e10),
            bijector=tfb.Reciprocal()
        ), 1)
    return j


def recip_norm_mat_layer(input, time_means, time_vars, name, n_infectious_compartments=1):
    '''Column Normalized Reciprical Gaussian trainable distribution. Zeros in starting matrix are preserved'''
    # add extra row that we use for concentration
    combined = np.stack((time_means, time_vars))
    indices = tf.cast(tf.where(time_means > 0), tf.int32)
    x = TrainableInputLayer(combined, constraint=MinMaxConstraint(
        1e-3, 1e10), name=name + '-hypers')(input)
    d = tfp.layers.DistributionLambda(lambda t: recip_norm_mat_dist(
        t[0, 0], t[0, 1], indices, sample_R=False, n_infectious_compartments=n_infectious_compartments), name=name + '-dist')(x)

    def reshaper(x, L=time_means.shape[0], indices=indices):
        if tf.rank(x) == 1:
            x = x[tf.newaxis, ...]
        mat = tf.map_fn(lambda v: tf.scatter_nd(indices, v, (L, L)), x)
        tmat_sample = tf.linalg.diag(
            1 - tf.reduce_sum(mat, axis=-1)) + mat
        # making sure sampled values are valid
        tf.debugging.assert_non_negative(tmat_sample,
                                         message='Sampled transition matrix seems to have negative values. If you have multiple infectious'
                                         ' compartments, please change n_infectious_compartments'
                                         ' based on your epidemiology model.')
        return tmat_sample
    return d, reshaper


def categorical_normal_layer(input, start_logits, start_mean, start_scale, pad, name, start_high=0.5):
    L = start_logits.shape[0]
    x = TrainableInputLayer(start_logits, name=name +
                            '-start-logit-hypers')(input)
    y = tf.keras.layers.Dense(2,
                              kernel_initializer=tf.constant_initializer(
                                  value=[start_mean, start_scale]),
                              name=name + '-norm-hypers',
                              use_bias=False,
                              dtype=x.dtype)(input)
    d = tfp.layers.DistributionLambda(lambda t:
                                      tfd.Blockwise(tfd.JointDistributionSequential([
                                          tfd.Multinomial(
                                              total_count=1,
                                              logits=t[0][0]),
                                          tfd.Sample(
                                              tfd.TruncatedNormal(
                                                  loc=tf.clip_by_value(
                                                      t[1][0, 0], 1e-3, start_high - 1e-3),
                                                  scale=1e-3 +
                                                  tf.math.sigmoid(t[1][0, 1]),
                                                  low=0.0,
                                                  high=start_high), sample_shape=[L]),
                                      ])), name=name + '-dist')([x, y])

    def reshaper(x):
        m = tf.squeeze(x[..., :L] * x[..., L:])
        return tf.stack([m] + pad * [tf.zeros_like(m)], axis=-1)
    return d, reshaper


def normal_mat_layer(input, start, name, start_var=1, clip_high=1e10):
    '''Normally distributed trainable distribution. Zeros in mobility matrix are preserved'''
    # stack variance
    start_val = np.concatenate((
        start[np.newaxis, ...],
        np.tile(start_var, start.shape)[np.newaxis, ...]
    ))
    indices = tf.cast(tf.where(start > 0), tf.int32)
    x = TrainableInputLayer(start_val, name=name + '-hypers',
                            constraint=PositiveMaskedConstraint(start_val > 0))(input)
    x = tf.keras.layers.Lambda(
        lambda x: x + 1e-10 * np.mean(start), name=name + '-jitter')(x)
    d = tfp.layers.DistributionLambda(lambda t: recip_norm_mat_dist(
        t[0, 0], t[0, 1], indices), name=name + '-dist')(x)

    def reshaper(x, L=start.shape[0], indices=indices):
        if tf.rank(x) == 1:
            x = x[tf.newaxis, ...]
        mat = tf.map_fn(lambda v: tf.scatter_nd(indices, v, (L, L)), x)
        return mat/tf.reduce_sum(mat, axis=-1, keepdims=True)
    return d, reshaper


class ParameterHypers:
    def __init__(self):
        self.beta_low = [0.01] * 3
        self.beta_high = [0.7] * 3
        self.beta_var = [0.05] *3
        self.start_high = 0.5
        self.start_var = 0.1
        self.R_var = 0.2
        self.beta_start = 0.1
        self.start_mean = 0.1
        self.start_scale = 0.1


class ParameterJoint(tf.keras.Model):
    def __init__(self, reshapers, **kwargs):
        '''Create trainable joint model for parameters'''
        self.reshapers = reshapers
        self.output_count = len(reshapers)
        super(ParameterJoint, self).__init__(**kwargs)

    def compile(self, optimizer, **kwargs):
        if 'loss' in kwargs:
            raise ValueError('Do not set loss')
        super(ParameterJoint, self).compile(
            optimizer, loss=self.output_count * [negloglik])

    def sample(self, N, return_joint=False):
        joint = self(tf.constant([1.]))
        if type(joint) != list:
            joint = [joint]
        y = [j.sample(N) for j in joint]
        v = [self.reshapers[i](s) for i, s in enumerate(y)]
        if return_joint:
            return v, y, joint
        else:
            return v


class MetaParameterJoint(ParameterJoint):
    def __init__(self, start_logits, mobility_matrix,
                 transition_matrix,
                 name='', hypers=None, n_infectious_compartments=1):
        '''Create trainable joint model for parameters'''
        if hypers is None:
            hypers = ParameterHypers()
        dense_layer_size = n_infectious_compartments
        beta_low = hypers.beta_low[0:dense_layer_size]
        beta_high = hypers.beta_high[0:dense_layer_size]
        beta_var = hypers.beta_var[0:dense_layer_size]
        i = tf.keras.layers.Input((1,))
        # infection parameter first
        beta_layer = tf.keras.layers.Dense(
            dense_layer_size,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(
                tf.math.log(hypers.beta_start)),
            name='beta')
        beta_dist = tfp.layers.DistributionLambda(
            lambda b: tfd.Independent(tfd.TruncatedNormal(
                loc=tf.clip_by_value(tf.math.sigmoid(
                    b[..., 0:n_infectious_compartments]), beta_low, beta_high),
                scale=beta_var,
                low=beta_low,
                high=beta_high), 1),
            name='beta-dist'
        )(beta_layer(i))
        R_dist = normal_mat_layer(
            i, mobility_matrix,  start_var=hypers.R_var, name='R-dist')
        T_dist = recip_norm_mat_layer(
            i, *transition_matrix.prior_matrix(), name='T-dist', n_infectious_compartments=n_infectious_compartments)
        start_dist = categorical_normal_layer(
            i, start_logits, hypers.start_mean, hypers.start_scale, len(transition_matrix.names) - 1, name='rho-dist', start_high=hypers.start_high)
        reshapers = [R_dist[1], T_dist[1], start_dist[1], lambda x: x]
        super(MetaParameterJoint, self).__init__(reshapers=reshapers, inputs=i, outputs=[
            R_dist[0], T_dist[0], start_dist[0], beta_dist], name=name + '-model')


class TrainableMetaModel(tf.keras.Model):
    def __init__(self, start, mobility_matrix, compartment_matrix, infect_func, timesteps, loss_fxn, populations=None):
        super(TrainableMetaModel, self).__init__()
        self.R_layer = TrainableInputLayer(mobility_matrix,
                                           constraint=NormalizationConstraint(1, mobility_matrix > 0))
        self.T_layer = TrainableInputLayer(
            compartment_matrix, NormalizationConstraint(1, compartment_matrix > 0))
        self.start_layer = TrainableInputLayer(
            start, MinMaxConstraint(0., 0.3))
        self.metapop_layer = MetapopLayer(timesteps, infect_func, populations)
        self.traj_layer = AddSusceptibleLayer(name='traj')
        self.agreement_layer = tf.keras.layers.Lambda(loss_fxn)

    def call(self, inputs):
        self.R = self.R_layer(inputs)
        self.T = self.T_layer(inputs)
        self.rho = self.start_layer(inputs)
        x = self.metapop_layer([self.R, self.T, self.rho])
        self.traj = self.traj_layer(x)
        self.agreement = self.agreement_layer(self.traj)
        return self.traj, self.agreement

    def get_traj(self):
        return self([0.0])[0]

    def compile(self, optimizer, **kwargs):
        if 'loss' in kwargs:
            raise ValueError(
                'Do not specificy loss, instead use loss_fxn in constructor')
        super(TrainableMetaModel, self).compile(
            optimizer, [None, 'mean_absolute_error'], **kwargs)

    def fit(self, steps=100, **kwargs):
        super(TrainableMetaModel, self).fit(
            steps * [0.], steps * [0.], batch_size=1, **kwargs)


class MetaModel(tf.keras.Model):
    def __init__(self, infect_func, timesteps, populations=None):
        super(MetaModel, self).__init__()
        self.metapop_layer = MetapopLayer(
            timesteps, infect_func, populations=populations)
        self.traj_layer = AddSusceptibleLayer(name='traj')

    def call(self, R, T, rho, params):
        # making sure R is normalized
        R /= tf.reduce_sum(R, axis=-1, keepdims=True)
        if tf.rank(R) == 2:
            R, T, rho = R[tf.newaxis, ...], T[tf.newaxis, ...], rho[tf.newaxis, ...]
        x = self.metapop_layer([R, T, rho, params])
        traj = self.traj_layer(x)
        return traj


class TrainableInputLayer(tf.keras.layers.Layer):
    ''' Create trainable input layer'''

    def __init__(self, initial_value, constraint=None, regularizer=None, **kwargs):
        super(TrainableInputLayer, self).__init__(**kwargs)
        flat = initial_value.flatten()
        self.initial_value = initial_value
        self.w = self.add_weight(
            'value',
            shape=initial_value.shape,
            initializer=tf.constant_initializer(flat),
            constraint=constraint,
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs):
        batch_dim = tf.shape(inputs)[:1]
        return tf.tile(self.w[tf.newaxis, ...], tf.concat((batch_dim, tf.ones(tf.rank(self.w), dtype=tf.int32)), axis=0))


class DeltaRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, value, strength=1e-3):
        self.strength = strength
        self.value = value

    def __call__(self, x):
        return self.strength * tf.reduce_sum((x - self.value)**2)


class MinMaxConstraint(tf.keras.constraints.Constraint):
    ''' Makes weights normalized after reshape and applying mask'''

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, w):
        return tf.clip_by_value(w, self.min, self.max)


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
        result = tf.concat((S[:, :, :, tf.newaxis], trajs), axis=-1)
        # want batch index first
        result = tf.transpose(result, perm=[1, 0, 2, 3])
        return result


class MetapopLayer(tf.keras.layers.Layer):
    def __init__(self, timesteps, infect_func, populations=None, dtype=tf.float64):
        super(MetapopLayer, self).__init__(dtype=dtype)
        self.infect_func = infect_func
        self.timesteps = timesteps
        self.populations = populations

    def build(self, input_shape):
        self.N, self.M, self.C = input_shape[2]
        if self.populations is None:
            self.populations = tf.ones((self.M, ), dtype=self.dtype)

    def call(self, inputs):
        R, T, rho0 = inputs[:3]
        infect_params = inputs[3:]
        trajs_array = tf.TensorArray(size=self.timesteps, element_shape=(
            self.N, self.M, self.C), dtype=self.dtype)

        def body(i, prev_rho, trajs_array):
            # write first so we get starting rho
            trajs_array = trajs_array.write(i, prev_rho)
            pop = tf.convert_to_tensor(
                self.populations, dtype=self.dtype)[tf.newaxis, ...]
            batched_pop = tf.repeat(pop, self.N, axis=0)
            # compute effective pops
            neff = batched_pop[:, tf.newaxis, :] @  R
            neff_compartments = tf.reshape(batched_pop, (-1, self.M, 1, 1)) * tf.reshape(prev_rho, (-1, self.M, 1, self.C)) *\
                R[..., tf.newaxis]

            neff = tf.squeeze(neff, axis=1)
            infect_prob = self.infect_func(
                neff_compartments, neff, * infect_params)
            # infect them
            new_infected = (1 - tf.reduce_sum(prev_rho, axis=-1)) * \
                tf.einsum('ijk,ik->ij', R, infect_prob)
            # create new compartment values
            rho = tf.einsum('ijk,ikl->ijl', prev_rho, T) + \
                new_infected[:, :, tf.newaxis] * tf.constant([1] +
                                                             [0 for _ in range(self.C - 1)], dtype=self.dtype)
            # project back to allowable values
            rho = tf.clip_by_value(rho, 0, 1)
            return i + 1, rho, trajs_array
        cond = lambda i, *_: i < self.timesteps
        _, _, trajs_array = tf.while_loop(
            cond, body, (0, rho0, trajs_array))
        return tf.cast(trajs_array.stack(), tf.float32)


class ContactInfectionLayer(tf.keras.layers.Layer):
    def __init__(self, initial_beta,  infectious_compartments, clip_low=0.01, clip_high=0.5, area=None, **kwargs):
        super(ContactInfectionLayer, self).__init__(**kwargs)
        self.infectious_compartments = infectious_compartments
        self.beta = tf.Variable(
            initial_value=tf.reshape(initial_beta, (-1,)),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, clip_low, clip_high),
            name='contact-beta',
        )
        self.beta = tf.cast(self.beta, tf.float64)
        self.area = area

    def call(self, neff_compartments, neff):
        fxn = contact_infection_func(
            self.infectious_compartments, area=self.area)
        p = fxn(neff_compartments, neff, self.beta)
        return p


def contact_infection_func(infectious_compartments, area=None, dtype=tf.float64):
    if area is not None:
        def density_fxn(n, area=area, xi=0.005):
            return 2. - tf.math.exp(-xi * n / area)
    else:
        def density_fxn(n):
            return tf.ones_like(n)

    def fxn(neff_compartments, neff,  beta, infectious_compartments=infectious_compartments):
        beta = tf.reshape(beta, (-1, len(infectious_compartments)))
        if neff_compartments.dtype != dtype:
            neff_compartments = tf.cast(neff_compartments, dtype=dtype)
        # k is the average number of contacts across the whole population
        ntot = tf.reduce_sum(neff, axis=1)
        k = 10.
        z = ntot * k / tf.reduce_sum(neff * density_fxn(neff), axis=1)
        p_getting_infected_in_patch_infectious_compartments = []
        for i, infected_compartment in enumerate(infectious_compartments):
            ninf = tf.zeros_like(neff_compartments[:, :, :, 0])
            ninf += neff_compartments[:, :, :, infected_compartment]
            p_getting_infected_in_patch = tf.math.exp(tf.math.log(1 - beta[:,i,tf.newaxis])
                                                      * density_fxn(neff) * z[..., tf.newaxis]
                                                      * tf.reduce_sum(ninf, axis=1) / neff)
            p_getting_infected_in_patch_infectious_compartments.append(
                p_getting_infected_in_patch)
        p_getting_infected_in_patch_infectious_compartments = tf.cast(
            p_getting_infected_in_patch_infectious_compartments, dtype=dtype)
        p = 1 - tf.reduce_prod(p_getting_infected_in_patch_infectious_compartments, axis=0)
        return p
    return fxn


def negloglik(y, rv_y):
    logp = rv_y.log_prob(y + EPS)
    logp = tf.reduce_sum(tf.reshape(logp, (tf.shape(y)[0], -1)), axis=1)
    return -logp
