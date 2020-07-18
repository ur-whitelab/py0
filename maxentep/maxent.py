import numpy as np
from scipy.special import softmax
import tensorflow as tf
from math import sqrt

class Prior:
    def expected(self, l):
        raise NotImplementedError()
    def expected_grad(self, l):
        raise NotImplementedError
    def log_denom(self, l):
        raise NotImplementedError

class EmptyPrior(Prior):
    def expected(self, l):
        return 0.0
    def expected_grad(self, l):
        return 0.0
    def log_denom(self, l):
        return 0.0

class Laplace(Prior):
    def __init__(self, sigma):
        self.sigma = sigma
    def expected(self, l):
        return -1. * l * self.sigma**2 / (1. - l**2 * self.sigma**2 / 2)
    def expected_grad(self, l):
        return (1.5 - 1./(l**2 * self.sigma**2))
    def log_denom(self, l):
        # cap it to stop stupid stuff
        return np.log(max(1e-8, 1. / (l + np.sqrt(2)/self.sigma) + 1. / (np.sqrt(2)/self.sigma - l)))

class Restraint:
    def __init__(self, fxn, target, prior):
        self.target = target
        self.fxn = fxn
        self.prior = prior

    def __call__(self, traj):
        return self.fxn(traj) - self.target

class AvgLayer(tf.keras.layers.Layer):
    def __init__(self, reweight_layer):
        super(AvgLayer, self).__init__()
        if type(reweight_layer) != ReweightLayer:
            raise TypeError()
        self.rl = reweight_layer
    def call(self, gk, weights):
        mask = tf.cast(tf.equal(self.rl.sigmas, 0), tf.float32)
        # sum over trajectories
        e_gk = tf.reduce_sum(gk * weights[:, tf.newaxis], axis=0)
        # add laplace term
        # cannot rely on mask due to no clip
        err_e_gk = e_gk + mask * (1.5 - tf.math.divide_no_nan(1.,(self.rl.l**2 * self.rl.sigmas**2)))
        return err_e_gk

class ReweightLayer(tf.keras.layers.Layer):
    def __init__(self, sigmas):
        super(ReweightLayer, self).__init__()
        l_init = tf.random_uniform_initializer(-10, 10)
        restraint_dim = len(sigmas)
        self.l = tf.Variable(
            initial_value=l_init(shape=(restraint_dim,), dtype='float32'),
            trainable=True,
            name='maxent-lambda'
        )
        self.sigmas = sigmas
    def call(self, gk):
        # add priors
        mask = tf.cast(tf.equal(self.sigmas, 0), tf.float32)
        two_sig = tf.math.divide_no_nan(sqrt(2), self.sigmas)
        prior_term = mask * tf.math.log(
            tf.clip_by_value(1. / (self.l + two_sig) + 1. / (two_sig - self.l),
            1e-8, 1e8))
        # sum-up constraint terms
        logits = tf.reduce_sum(-self.l[tf.newaxis, :] * gk + prior_term[tf.newaxis, :], axis=1)
        # compute per-trajectory weights
        weights = tf.math.softmax(logits)
        return weights

def _compute_restraints(trajs, restraints):
    N = trajs.shape[0]
    K = len(restraints)
    gk = np.empty((N, K))
    for i in range(N):
        gk[i, :] = [r(trajs[i]) for r in restraints]
    return gk

class MaxentModel(tf.keras.Model):
    def __init__(self, restraints, use_cov=False, name='maxent-model', **kwargs):
        super(MaxentModel, self).__init__(name=name, **kwargs)
        self.restraints = restraints
        restraint_dim = len(restraints)
        # identify prior
        prior = type(restraints[0].prior)
        if prior != Laplace:
            raise NotImplementedError()
        # double-check
        for r in restraints:
            if type(r.prior) != prior:
                raise ValueError('Can only do restraints of one type')
        sigmas = np.array([r.prior.sigma for r in restraints], dtype=np.float32)
        self.weight_layer = ReweightLayer(sigmas)
        self.avg_layer = AvgLayer(self.weight_layer)
        self.lambdas = self.weight_layer.l
        self.prior = prior
    def call(self, inputs):
        weights = self.weight_layer(inputs)
        wgk = self.avg_layer(inputs, weights)
        return [weights, wgk]
    def compile(self, optimizer='rmsprop', loss=None, **kwargs):
        return super(MaxentModel, self).compile(optimizer, loss=[None, loss], **kwargs)
    def fit(self, trajs, batch_size=16, **kwargs):
        gk = _compute_restraints(trajs, self.restraints)
        inputs = gk.astype(np.float32)
        data = tf.data.Dataset.from_tensor_slices((inputs, np.zeros_like(gk,dtype=np.float32)))
        data = data.shuffle(batch_size * 4).batch(batch_size)
        result = super(MaxentModel, self).fit(data, **kwargs)
        self.traj_weights = self.call(inputs)[0]
        return result
