import numpy as np
from scipy.special import softmax

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

def reweight(trajs, restraints, iter=100, batch_size=32, learning_rate=1e-2, callback=None):
    '''callback is called with f(iteration_number, weights, exp(restraint - target), agreement)
    '''
    N = trajs.shape[0]
    K = len(restraints)
    lambdas = np.random.uniform(-1, 1, size=K)
    # add a tiny bit for possible zero-gradients
    accum_queue = np.zeros((K,)) + 1e-12

    # get value of all restraints on trajectories
    # note wish I could do this without for loop comprehension
    # guess it's a one time loop
    gk = np.empty((N, K))
    for i in range(N):
        gk[i, :] = [r(trajs[i]) for r in restraints]

    for i in range(iter):
        # update weights
        # compute effect of priors
        prior_term = np.array([r.prior.log_denom(l) for l,r in zip(lambdas, restraints)]).reshape(1, K)
        # sum-up constraint terms
        logits = np.sum(-lambdas.reshape(1, K) * gk + prior_term, axis=1)
        # compute per-trajectory weights
        weights = softmax(logits)
        # make w be N x 1 to broadcast
        # sum over N
        e_gk = np.sum(gk * weights[:, np.newaxis], axis=0)
        e_g2k = np.sum(gk**2 * weights[:, np.newaxis], axis=0)
        # get two gradient terms
        agreement = (e_gk + [r.prior.expected(l) for l,r in zip(lambdas, restraints)])
        scale = ([r.prior.expected_grad(l) for l,r in zip(lambdas, restraints)] + e_gk**2 - e_g2k)
        # compute gradient and penalty (?)
        grad = agreement * scale
        # add noise
        if K > 1:
            grad *= np.random.randint(0,2, size=K)
        # compute grad accumulation
        accum_queue += grad**2
        # update with adagrad-ish
        lambdas -= grad / np.sqrt(accum_queue) * learning_rate
        # callback
        if callback is not None:
            callback(i, weights, lambdas, e_gk, agreement, scale)
    return weights

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
