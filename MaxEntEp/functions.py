import numpy as np
import matplotlib.pyplot as plt



def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def step(compartments, params):
  E, A, I, R = compartments[0], compartments[1], compartments[2], compartments[3]
  alpha, eta, gamma, mu, pi, beta = params['alpha'], params['eta'],\
      params['gamma'], params['mu'],\
      params['pi'], params['beta']
  E_next = min(1., max(0., (1. - eta) * E + (1. - A - I - R - E)
                       * pi * beta))  # whatever \Pi_i(t) is... Assuming
  A_next = min(1., max(0., eta * E + (1 - alpha) * A))
  I_next = min(1., max(0., alpha * A + (1 - mu) * (1 - gamma) * I))
  R_next = min(1., max(0., R + (1 - (1 - mu) * (1 - gamma)) * I))
  return [E_next, A_next, I_next, R_next]


