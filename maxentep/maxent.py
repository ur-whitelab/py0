import numpy as np
from scipy.special import softmax

class Prior:
    def expected(self, l):
        raise NotImplementedError()
    def expected_grad(self, l):
        raise NotImplementedError
    def log_denom(self, l):
        raise NotImplementedError

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

def reweight(trajs, restraints, iter=100, batch_size=32, learning_rate=1e-2, callback=None, A=25):
    '''callback is called with f(iteration_number, weights, exp(restraint - target), agreement)
    '''
    N = trajs.shape[0]
    K = len(restraints)
    lambdas = np.random.uniform(-1, 1, size=K)
    # add a tiny bit for possible zero-gradients
    accum_queue = np.zeros((A, K)) + 1e-12
    accum_ptr = 0

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
        # compute gradient
        grad = agreement * scale
        # add noise
        if K > 1:
            grad *= np.random.randint(0,2, size=K)
        # compute grad accumulation
        accum_queue[accum_ptr] = grad**2
        accum_ptr = (accum_ptr + 1) % A
        # update with adagrad
        lambdas -= grad / np.sqrt(np.sum(accum_queue, axis=0)) * learning_rate
        # callback
        if callback is not None:
            callback(i, weights, lambdas, e_gk, agreement, scale)
    return weights

def reweight_laplace(trajs, restraints, **kw_args):
    '''Assumes laplace prior, fixed time restraints. Restraints
    should be an K x (2 + T)... array. K is number of restraints.
    The first two values in the second dimension are target value and
    Laplace prior sigma. T is the dimension of trajs - 1 used to index trajectory.
    '''
    restraints = np.array(restraints)
    if len(restraints.shape) == 1 or restraints.shape[1] != 2 + len(trajs.shape[1:]):
        raise ValueError('Bad restraint shape')
    _restraints = [Restraint(lambda t: t[tuple(i.astype(np.int))], v, Laplace(s)) for\
                    i,v,s in zip(restraints[:, 2:],
                                 restraints[:, 0],
                                 restraints[:, 1])]
    return reweight(trajs, _restraints, **kw_args), _restraints
