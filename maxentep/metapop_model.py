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


def recip_norm_mat_dist(trans_times, trans_times_var, indices):
    values = tf.gather_nd(trans_times, indices)
    v_vars = tf.gather_nd(trans_times_var, indices)
    j = tfd.Independent(tfd.TransformedDistribution(
        tfd.TruncatedNormal(loc=tf.clip_by_value(
            values, 1 + 1e-3, 1e10), scale=v_vars, low=1., high=1e10),
        bijector=tfb.Reciprocal()
    ), 1)
    return j


def recip_norm_mat_layer(input, time_means, time_vars, name):
    '''Column Normalized Reciprical Gaussian trainable distribution. Zeros in starting matrix are preserved'''
    # add extra row that we use for concentration
    combined = np.stack((time_means, time_vars))
    indices = tf.cast(tf.where(time_means > 0), tf.int32)
    x = TrainableInputLayer(combined, constraint=MinMaxConstraint(
        1e-3, 1e10), name=name + '-hypers')(input)
    d = tfp.layers.DistributionLambda(lambda t: recip_norm_mat_dist(
        t[0, 0], t[0, 1], indices), name=name + '-dist')(x)

    def reshaper(x, L=time_means.shape[0], indices=indices):
        if tf.rank(x) == 1:
            x = x[tf.newaxis, ...]
        mat = tf.map_fn(lambda v: tf.scatter_nd(indices, v, (L, L)), x)
        return tf.linalg.diag(1 - tf.reduce_sum(mat, axis=-1)) + mat
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
    '''Normally distributed trainable distribution. Zeros in starting matrix are preserved'''
    # stack variance
    start_val = np.concatenate((
        start[np.newaxis, ...],
        np.tile(start_var, start.shape)[np.newaxis, ...]
    ))
    # zero-out variance of zeroed starts
    start_val[1] = start_val[1] * (start_val[0] > 0)
    x = TrainableInputLayer(start_val, name=name + '-hypers',
                            constraint=PositiveMaskedConstraint(start_val > 0))(input)
    x = tf.keras.layers.Lambda(
        lambda x: x + 1e-10 * np.mean(start), name=name + '-jitter')(x)
    return tfp.layers.DistributionLambda(
        lambda t: tfd.Independent(tfd.TruncatedNormal(
            loc=t[0, 0],
            low=0.0,
            high=clip_high,
            scale=1e-3 + tf.math.sigmoid(t[0, 1])), 2),
        name=name + '-dist')(x), lambda x: x/tf.reduce_sum(x, axis=-1, keepdims=True)#[..., tf.newaxis]


class ParameterHypers:
    def __init__(self):
        self.beta_low = 0.01
        self.beta_high = 0.7
        self.beta_var = 0.05
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
                 name='', hypers=None):
        '''Create trainable joint model for parameters'''
        if hypers is None:
            hypers = ParameterHypers()
        i = tf.keras.layers.Input((1,))
        # infection parameter first
        beta_layer = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(
                tf.math.log(hypers.beta_start)),
            name='beta')
        beta_dist = tfp.layers.DistributionLambda(
            lambda b: tfd.Independent(tfd.TruncatedNormal(
                loc=tf.clip_by_value(tf.math.sigmoid(
                    b[..., 0]), hypers.beta_low + 1e-3, hypers.beta_high - 1e-3),
                scale=hypers.beta_var,
                low=hypers.beta_low,
                high=hypers.beta_high), 1),
            name='beta-dist'
        )(beta_layer(i))
        R_dist = normal_mat_layer(
            i, mobility_matrix,  start_var=hypers.R_var, name='R-dist')
        T_dist = recip_norm_mat_layer(
            i, *transition_matrix.prior_matrix(), name='T-dist')
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
        #making sure R is normalized
        R /= tf.reduce_sum(R, axis=-1 , keepdims=True)
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
        # else:
        #     eff_pop0 = tf.convert_to_tensor(
        #         self.populations, dtype=self.dtype)[tf.newaxis, ...]
        #     self.eff_pop0 = tf.repeat(eff_pop0, self.N, axis=0)
        #     # print('eff_pop0 shape', eff_pop0.shape)

    def call(self, inputs):
        R, T, rho0 = inputs[:3]
        # infect_params = [tf.cast(_, dtype=self.dtype) for _ in inputs[3:]]
        infect_params = inputs[3:]
        trajs_array = tf.TensorArray(size=self.timesteps, element_shape=(
            self.N, self.M, self.C), dtype=self.dtype)

        def body(i, prev_rho, trajs_array):
            # print(i)
            # print('T', T)
            # pre_eff_pop = tf.reshape(pre_eff_pop, (1, self.M))
            # write first so we get starting rho
            trajs_array = trajs_array.write(i, prev_rho)
            pop = tf.convert_to_tensor(
                self.populations, dtype=self.dtype)[tf.newaxis, ...]
            batched_pop = tf.repeat(pop, self.N, axis=0)
            
            # compute effective pops
            # print(
            #     'self.eff_pop0[tf.newaxis, :, tf.newaxis, tf.newaxis] shape', self.eff_pop0.shape)
            neff = batched_pop[:, tf.newaxis, :] @  R
            # print('neff shape', neff.shape)
            
            # neff_compartments = tf.reshape(batched_pop, (-1, self.M, 1, 1)) * tf.reshape(prev_rho, (-1, self.M, 1, self.C)) *\
            #     tf.reshape(tf.transpose(
            #         R, perm=[0, 2, 1]), (-1, self.M, self.M, 1))
            # neff_compartments = tf.tensordot(neff[..., tf.newaxis], prev_rho, axes = 1)  # [..., tf.newaxis, tf.newaxis]
            # almost match
            # neff_compartments = self.eff_pop0[:, :, tf.newaxis, tf.newaxis] * \
            #     prev_rho[:, :, tf.newaxis, :] * R[..., tf.newaxis]

            # #new
            # neff_compartments = self.eff_pop0[:, tf.newaxis, :, tf.newaxis] * \
            #     tf.reshape(prev_rho, (-1, self.M, 1, self.C)) * \
            #     R[..., tf.newaxis]
            
            neff_compartments = tf.reshape(batched_pop, (-1, self.M, 1, 1)) * tf.reshape(prev_rho, (-1, self.M, 1, self.C)) *\
                                    R[..., tf.newaxis]
                                
            neff = tf.squeeze(neff, axis=1)
            # print('R shape', R.shape)
            # tf.print('neff_compartments ', neff_compartments[0,2,:], summarize=-1)
            # neff = self.eff_pop0[tf.newaxis, :] * tf.reduce_sum(R, axis=2)
            # ntot =tf.reduce_sum(ntot)
            # compute infected prob
            # infect_params = [tf.cast(_,dtype=self.dtype) for _ in infect_params]
            # tf.print('infect_params', [n.dtype for n in infect_params])
            infect_prob = self.infect_func(
                neff_compartments, neff, * infect_params)
            # tf.print('infect_prob', infect_prob, summarize=-1)
            # infect them
            new_infected = (1 - tf.reduce_sum(prev_rho, axis=-1)) * \
                tf.einsum('ijk,ik->ij', R, infect_prob)
            # tf.print('old new_infected shape', new_infected.shape)
            # tf.print(' shape before', (1 - tf.reduce_sum(prev_rho, axis=-1)).shape)
            # new_infected = tf.reshape((1 - tf.reduce_sum(prev_rho, axis=-1)), (-1, 1, self.M)) * \
            #                         tf.reshape(infect_prob,(-1, 1, self.M)) @ tf.transpose(R, perm=[0, 2, 1])
            # # # tf.print(' new_infected shape', new_infected.shape)
            # new_infected = tf.squeeze(new_infected, axis=1)
            
            # create new compartment values
            rho = tf.einsum('ijk,ikl->ijl', prev_rho, T) + \
                new_infected[:, :, tf.newaxis] * tf.constant([1] +
                                                             [0 for _ in range(self.C - 1)], dtype=self.dtype)
            # project back to allowable values
            rho = tf.clip_by_value(rho, 0, 1)
            # tf.print('rho: ',
                    #  rho.shape, summarize=-1)
            # tf.print('rho :', rho)
            #rho /= tf.clip_by_value(tf.reduce_sum(rho, axis=-1, keepdims=True), 1, 1000000)
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
        fxn = contact_infection_func(self.infectious_compartments, area =self.area)
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
        if neff_compartments.dtype != dtype:
            neff_compartments = tf.cast(neff_compartments, dtype= dtype)
        ninf = tf.zeros_like(neff_compartments[:, :, :, 0])
        # ninf.shape is [batch_size, M, M]
        
        # k is the average number of contacts across the whole population
        ntot = tf.reduce_sum(neff, axis =1)
        # tf.print('ntot', ntot)
        k = 10.
        z = ntot * k / tf.reduce_sum(neff * density_fxn(neff), axis=1)
        for i in infectious_compartments:
            ninf += neff_compartments[:, :, :, i]
        # tf.print('density_fxn', density_fxn(neff), summarize=-1)
        # tf.print('ninf', ninf, summarize=-1)
        # tf.print('z ', z, summarize=-1)
        # tf.print('sum axis 1', tf.reduce_sum(ninf, axis=1), summarize=-1)
        # getting infected and asymptomatic in double percision
        # sum_inf = tfp.math.reduce_kahan_sum(ninf, axis=1)
        # sum_inf = sum_inf.total - sum_inf.correction
        # tf.print('sum_inf', sum_inf, summarize=-1)

        p = 1 - tf.math.exp(tf.math.log(1 - tf.reshape(beta, (-1, 1)))
                            * density_fxn(neff) * z[..., tf.newaxis] 
                            * tf.reduce_sum(ninf, axis=1) / neff)

        # p = 1 - tf.math.exp(tf.math.log(1 - tf.reshape(beta, (-1, 1)))
        #                             * density_fxn(neff) * z[..., tf.newaxis]
        #                             * tf.math.exp(tf.math.log(tf.reduce_sum(ninf * 1e3, axis=1)) + tf.math.log(1/neff * 1e3) - tf.math.log(1e6)))
        # tf.print(p, summarize=-1)
        # p = 1. - tf.math.pow(1. - tf.reshape(beta, (-1, 1)),  density_fxn(neff) * z *
        #                     tf.reduce_sum(ninf, axis=1) / neff)
        # tf.print('P_0',   1. - tf.pow(1. - tf.reshape(beta, (-1, 1))[0,0],  density_fxn(neff)[0,0] * tf.reshape(z, (-1, 1))[0,0] * \
        #                                tf.math.divide_no_nan(tf.math.reduce_sum(ninf, axis=1)[0, 0], neff[0, 0])), summarize=-1)

        # p = 1 - tf.reduce_prod(tf.math.exp((tf.math.log(1 - tf.reshape(beta, (-1, 1)))
        #                     * density_fxn(neff) * z[..., tf.newaxis]
        #                     * ninf/ neff)), axis =2)
        return p
    return fxn



## Mine

# def contact_infection_func(infectious_compartments, area=None, noise=None):
#     if area is not None:
#         def density_fxn(n, area=area, xi=0.005):
#             return 2 - tf.math.exp(-xi * n / area)
#     else:
#         def density_fxn(n):
#             return tf.ones_like(n)

#     def fxn(neff_compartments, neff, beta, infectious_compartments=infectious_compartments):
#         ninf = tf.zeros(tf.shape(neff_compartments)[:-1])
#         ntot = tf.reduce_sum(neff, axis=1)
#         # k is the average number of contacts across the whole population
#         k = 10
#         z = ntot * k / tf.reduce_sum(neff * density_fxn(neff), axis=1)
#         for i in infectious_compartments:
#             ninf += neff_compartments[:, :, i]
#         # tf. print('neff shape', neff.shape, summarize=-1)
#         # tf.print('ninf', ninf, summarize=-1)
#         # tf.print('ntot', ntot,  summarize=-1)
#         p = 1 - tf.math.exp(tf.math.log(1 - tf.reshape(beta, (-1, 1)))
#                             * density_fxn(neff) * z[..., tf.newaxis]
#                             * tf.math.divide_no_nan(ninf, neff))
#         # tf.print('z', z)
#         if noise:
#             # adding noise
#             noised_p = p *\
#                 (tf.random.uniform(
#                     shape=tf.shape(p), minval=1.0 - noise, maxval=1.0 + noise))
#             return noised_p
#         else:
#             return p
#     return fxn



def negloglik(y, rv_y):
    logp = rv_y.log_prob(y + EPS)
    logp = tf.reduce_sum(tf.reshape(logp, (tf.shape(y)[0], -1)), axis=1)
    return -logp


## Mine
# class MetapopLayer(tf.keras.layers.Layer):
#     def __init__(self, timesteps, infect_func, populations=None, dtype=tf.float32):
#         super(MetapopLayer, self).__init__(dtype=dtype)
#         self.infect_func = infect_func
#         self.timesteps = timesteps
#         self.populations = populations

#     def build(self, input_shape):
#         self.N, self.M, self.C = input_shape[2]
#         if self.populations is None:
#             self.populations = tf.ones((self.M, ))

#     def call(self, inputs):
#         R, T, rho0 = inputs[:3]
#         infect_params = inputs[3:]
#         trajs_array = tf.TensorArray(size=self.timesteps, element_shape=(
#             self.N, self.M, self.C), dtype=self.dtype)
#         eff_pop0 = tf.convert_to_tensor(
#             self.populations, dtype=self.dtype)[tf.newaxis, tf.newaxis, :]
#         eff_pop0 = tf.repeat(eff_pop0, self.N, axis=0)

#         def body(i, prev_rho, trajs_array):
#             # print(i)
#             # print('T', T)
#             # pre_eff_pop = tf.reshape(pre_eff_pop, (1, self.M))
#             # write first so we get starting rho
#             trajs_array = trajs_array.write(i, prev_rho)
#             # pop = tf.convert_to_tensor(
#             #     self.populations, dtype=self.dtype)
#             # neff = tf.reduce_sum(pre_eff_pop[tf.newaxis, :] * tf.reshape(
#             #     tf.transpose(R), (-1, self.M, self.M)), axis = 2)
#             # neff = pre_eff_pop[tf.newaxis, :] * tf.reduce_sum(R, axis=1)
#             # print(i)
#             # tf.print('pre_eff_pop', pre_eff_pop, summarize=-1)
#             # tf.print('R :',R,summarize=-1)
#             # tf.print('and')
#             # tf.print('R.T', tf.transpose(R, perm=[0, 2, 1]) , summarize=-1)
#             # tf.print('tf.transpose(R) shape',
#             #          tf.transpose(R, perm=[0, 2, 1]).shape, summarize=-1)
#             # neff = tf.tensordot(pre_eff_pop,  tf.transpose(R, perm=[0, 2, 1]), axes = [[0,1],[1]])

#             neff = eff_pop0 @  R
#             # neff = tf.matmul(pre_eff_pop, tf.transpose(R, perm=[0, 2, 1]))
#             # neff = tf.tensordot(pre_eff_pop, tf.transpose(R, perm=[0, 2, 1]), axes =((0), (1,0)))
#             # neff = tf.einsum('ij,ijj->ij', pre_eff_pop,
#             #                  tf.transpose(R, perm=[0, 2, 1]))
#             # neff = tf.matmul(pre_eff_pop[tf.newaxis, tf.newaxis, :], tf.reshape(
#             #     R, (-1, self.M, self.M)))
#             neff = tf.squeeze(neff, axis=1)
#             # neff = eff_pop0 * \
#             #     tf.reshape(tf.transpose(
#             #         R, perm=[0, 2, 1]), (-1, self.M, self.M))
#             # tf.print('neff', neff.shape, summarize=-1)
#             ntot = tf.reduce_sum(neff, axis=1)
#             # tf.print('ntot', ntot, summarize=-1)
#             # neff = tf.squeeze(neff)

#             # neff = tf.reshape(neff, (-1, self.M))
#             # neff = tf.reduce_sum(pop[ :, tf.newaxis] *\
#             #     tf.reshape(tf.transpose(R), (-1, self.M, self.M)), axis = 1)
#             # tf.print('neff_body: ', neff.shape, summarize=-1)

#             # tf.print('neff', neff.shape, summarize=-1)
#             # tf.print('eff_pop0', tf.squeeze(eff_pop0, axis=0).shape)
#             # pop0 = tf.squeeze(eff_pop0, axis=0)
#             neff_compartments = prev_rho * \
#                 neff[..., tf.newaxis]  # pop0[..., tf.newaxis]  #
#             # tf.print('neff_compartments_body shape: ',
#             #          neff_compartments.shape, summarize=-1)
#             # compute infected prob
#             infect_prob = self.infect_func(
#                 neff_compartments, neff, *infect_params)
#             tf.print('infect_prob ', infect_prob, summarize=-1)
#             # infect them
#             new_infected = (1 - tf.reduce_sum(prev_rho, axis=-1)) * \
#                 tf.einsum('ijk,ik->ij', R, infect_prob)
#             # tf.print('new_infected old shape', new_infected.shape)
#             # new_infected = (1 - tf.reduce_sum(prev_rho, axis=-1)
#             #      )[:, tf.newaxis, :] * R @ infect_prob[..., tf.newaxis]

#             # new_infected = (1 - tf.reduce_sum(prev_rho, axis=-1)) * \
#             #                             infect_prob @ tf.transpose(R, perm=[0, 2, 1])
#             # tf.print('new_infected new shape', new_infected.shape)
#             # new_infected = tf.squeeze(new_infected, axis=-1)
#             # tf.print('bfore', (1 - tf.reduce_sum(prev_rho, axis=-1)).shape)
#             # tf.print('tf.einsum(ijk, ik -> ij, R, infect_prob)',
#             #          tf.einsum('ijk,ik->ij', R, infect_prob).shape)
#             # tf.print('R @ tf.transpose(infect_prob)',
#             #          (R @ tf.transpose(infect_prob)).shape)
#             # a = (1 - tf.reduce_sum(prev_rho, axis=-1))[:,tf.newaxis,:] * R @ tf.transpose(infect_prob)

#             # new_infected = tf.squeeze(new_infected, axis=-1)
#             # tf.print('new_infected', new_infected.shape, summarize=-1)

#             # tf.print('T', T.shape, summarize=-1)
#             new_infected = tf.clip_by_value(new_infected, 0, 1e10)
#             # create new compartment values
#             rho = tf.einsum('ijk,ikl->ijl', prev_rho, T) + \
#                 new_infected[:, :, tf.newaxis] * tf.constant([1] +
#                                                              [0 for _ in range(self.C - 1)], dtype=self.dtype)
#             # tf.print('rho ', rho, summarize=-1)
#             # project back to allowable values
#             # rho = tf.clip_by_value(rho, 0, 1)
#             #rho /= tf.clip_by_value(tf.reduce_sum(rho, axis=-1, keepdims=True), 1, 1000000)
#             return i + 1, rho, trajs_array
#         cond = lambda i, *_: i < self.timesteps
#         _, _, trajs_array = tf.while_loop(
#             cond, body, (0, rho0, trajs_array))
#         return trajs_array.stack()
