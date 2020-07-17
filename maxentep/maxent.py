import numpy as np
from scipy.special import softmax
import tensorflow as tf

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

# TODO make laplace maxent layer and add that to call
class MaxentLayer(tf.keras.layers.Layer):
    def __init__(self, restraint_dim):
        super(MaxentLayer, self).__init__()
        l_init = tf.random_uniform_initializer(-10, 10)
        self.l = tf.Variable(
            initial_value=l_init(shape=(restraint_dim,), dtype='float32'),
            trainable=True,
            name='maxent-lambda'
        )
    def call(self, gk):
        e_gk = _reweight_op2(self.l, gk)
        return e_gk

class ReweightLayer(tf.keras.layers.Layer):
    def __init__(self, maxent_layer):
        super(ReweightLayer, self).__init__()
        if type(maxent_layer) != MaxentLayer:
            raise TypeError('arg maxent_layer needs to be MaxentLayer')
        self.l = maxent_layer.l
    def call(self, gk):
        # sum-up constraint terms
        logits = tf.reduce_sum(-self.l[tf.newaxis, :] * gk, axis=1)
        # compute per-trajectory weights
        weights = tf.math.softmax(logits)
        return weights

@tf.custom_gradient
def _reweight_op(lambdas, gk):
    # sum-up constraint terms
    logits = tf.reduce_sum(-lambdas[tf.newaxis, :] * gk, axis=1)
    # compute per-trajectory weights
    weights = tf.math.softmax(logits)
    # sum over trajectories
    e_gk = tf.reduce_sum(gk * weights[:, tf.newaxis], axis=0)
    def grad_fn(dy):
        e_g2k = tf.reduce_sum(gk**2 * weights[:, tf.newaxis], axis=0)
        grad = -(e_gk**2 - e_g2k)
        # other gradients are undefined
        return grad * dy, None
    return e_gk, grad_fn

def _reweight_op2(lambdas, gk):
    # sum-up constraint terms
    logits = tf.reduce_sum(-lambdas[tf.newaxis, :] * gk, axis=1)
    # compute per-trajectory weights
    weights = tf.math.softmax(logits)
    # sum over trajectories
    e_gk = tf.reduce_sum(gk * weights[:, tf.newaxis], axis=0)
    return e_gk

def _compute_restraints(trajs, restraints):
    N = trajs.shape[0]
    K = len(restraints)
    gk = np.empty((N, K))
    for i in range(N):
        gk[i, :] = [r(trajs[i]) for r in restraints]
    return gk

class MaxentModel(tf.keras.Model):
    def __init__(self, restraint_dim, name='maxent-model', **kwargs):
        super(MaxentModel, self).__init__(name=name, **kwargs)
        self.me_layer = MaxentLayer(restraint_dim)
        self.weight_layer = ReweightLayer(self.me_layer)
        self.lambdas = self.me_layer.l
    def call(self, inputs):
        wgk = self.me_layer(inputs)
        weights = self.weight_layer(inputs)
        return [weights, wgk]
    def compile(self, optimizer='rmsprop', loss=None, **kwargs):
        return super(MaxentModel, self).compile(optimizer, loss=[None, loss], **kwargs)

    def fit(self, trajs, restraints, batch_size=32, **kwargs):
        gk = _compute_restraints(trajs, restraints)
        data = tf.data.Dataset.from_tensor_slices((gk.astype(np.float32), np.zeros_like(gk,dtype=np.float32))).shuffle(batch_size * 4).batch(batch_size)
        result = super(MaxentModel, self).fit(data, **kwargs)
        self.traj_weights = self.call(gk)[0]
        return result


def reweight_laplace(trajs, restraints, **kw_args):
    '''Assumes laplace prior, fixed time restraints. Restraints
    should be an K x (2 + T)... array. K is number of restraints.
    The first two values in the second dimension are target value and
    Laplace prior sigma. T is the dimension of trajs - 1 used to index trajectory.

    Example 1: [
        [0.5, 0.1, 10, 1]
        [0.6, 0.0, 25, 0]
    ]
    2 restraints corresponding to slice [10, 1] and [25, 0] in the trajectories.
    The first should be 0.5 with uncertainty (with Laplace dist) 0.1. The second
    is an exact restraint (no uncertainty) with value 0.6.
    '''
    restraints = np.array(restraints)
    if len(restraints.shape) == 1 or restraints.shape[1] != 2 + len(trajs.shape[1:]):
        raise ValueError('Bad restraint shape')

    # build restraints using uncertainties
    _restraints = []
    K = restraints.shape[0]

    for i in range(K):
        traj_index = tuple(restraints[i, 2:].astype(np.int))
        value = restraints[i, 0]
        uncertainty = restraints[i, 1]
        p = Laplace(uncertainty) if uncertainty > 0 else EmptyPrior()
        r = Restraint(lambda traj: traj[traj_index], value, p)
        _restraints.append(r)
    return reweight(trajs, _restraints, **kw_args), _restraints
