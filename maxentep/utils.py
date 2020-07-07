import numpy as np
import matplotlib.pyplot as plt


def _weighted_quantile(values, quantiles, sample_weight=None,
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


def traj_quantile(trajs, weights = None, names=None, plot_means=True, alpha=0.2):
    '''Make a plot of all the trajectories and the average trajectory based on
      parameter weights.'''

    if names is None:
        names = [f'Compartment {i}' for i in range(trajs.shape[-1])]
    if weights is None:
        w = np.ones(trajs.shape[0])
    else:
        w = weights
    w /= np.sum(w)

    x = range(trajs.shape[1])

    # weighted quantiles doesn't support axis
    # fake it using apply_along
    qtrajs = np.apply_along_axis(lambda x: _weighted_quantile(
        x, [1/3, 1/2, 2/3], sample_weight=w), 0, trajs)
    if plot_means:
        # approximate quantiles as distance from median applied to mean
        mtrajs = np.sum(trajs * w[:, np.newaxis, np.newaxis], axis=0)
        qtrajs[0, :, :] = qtrajs[0, :, :] - qtrajs[1, :, :] + mtrajs
        qtrajs[2, :, :] = qtrajs[2, :, :] - qtrajs[1, :, :] + mtrajs
        qtrajs[1, :, :] = mtrajs

    plt.xlabel('Timestep')
    plt.ylabel('Fraction of Population')
    for i in range(trajs.shape[-1]):
        plt.plot(x, qtrajs[1, :, i], color=f'C{i}', label=f'Compartment {names[i]}')
        plt.fill_between(x, qtrajs[0, :, i], qtrajs[-1, :, i],
                         color=f'C{i}', alpha=alpha)
    if not plot_means:
        plt.plot(x, np.sum(qtrajs[1, :, :], axis=1),
             color='gray', label='Total', linestyle=':')
    # add margin for legend
    plt.xlim(0, max(x) * 1.2)
    plt.legend(loc='center right')
