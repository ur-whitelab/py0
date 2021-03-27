import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import maxentep
import tensorflow as tf


class TransitionMatrix:
    def __init__(self, compartment_names, infectious_compartments):
        self.names = compartment_names
        self.infectious_compartments = infectious_compartments
        self.transitions = []
        self.mat = None

    def add_transition(self, name1, name2, time, time_var):
        if name1 not in self.names or name2 not in self.names:
            raise ValueError('name not in compartment names')
        if name1 == name2:
            raise ValueError('self-loops are added automatically')
        self.transitions.append([name1, name2, time, time_var])
        self.mat = None

    def prior_matrix(self):
        C = len(self.names)
        T1, T2 = np.zeros((C, C)), np.zeros((C, C))
        for n1, n2, v, vv in self.transitions:
            i = self.names.index(n1)
            j = self.names.index(n2)
            T1[i, j] = v
            T2[i, j] = vv
        return T1, T2

    def _make_matrix(self):

        C = len(self.names)
        T = np.zeros((C, C))
        for n1, n2, v, vv in self.transitions:
            i = self.names.index(n1)
            j = self.names.index(n2)
            T[i, j] = 1 / v
            # get what leaves
        np.fill_diagonal(T, 1 - np.sum(T, axis=1))
        self.mat = T

    @property
    def value(self):
        '''Return matrix value
        '''
        if self.mat is None:
            self._make_matrix()
        return self.mat


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


def patch_quantile(trajs, *args, figsize=(18, 18), patch_names=None, ** kw_args):
    '''does traj_quantile for trajectories of shape [ntrajs, time, patches, compartments]
    '''
    NP = trajs.shape[2]
    nrow = int(np.floor(np.sqrt(NP)))
    ncol = int(np.ceil(NP / nrow))
    print(f'Plotting {NP} patches in a {nrow} x {ncol} grid')
    fig, ax = plt.subplots(nrow, ncol, sharex=True,
                           sharey=True, figsize=figsize)
    for i in range(nrow):
        for j in range(ncol):
            if i * ncol + j == NP:
                break
            traj_quantile(trajs[:, :, i * ncol + j, :], *args, ax=ax[i, j],
                          add_legend=i == 0 and j == ncol - 1, **kw_args)
            ax[i, j].set_ylim(0, 1)
            if patch_names is None:
                ax[i, j].text(trajs.shape[1] // 2, 0.8,
                              f'Patch {i * ncol + j}')
            else:
                patch_names = patch_names
                ax[i, j].set_title(patch_names[i * ncol + j])

            if j == 0 and i == nrow // 2:
                ax[i, j].set_ylabel('Fraction')
            if i == nrow - 1 and j == ncol // 2:
                ax[i, j].set_xlabel('Time')
            if j >= NP % ncol:
                ax[nrow-1, j].set_visible(False)
        
    plt.tight_layout()


def traj_quantile(trajs, weights=None, figsize=(9, 9), names=None, plot_means=True, ax=None, add_legend=True, add_title=None, alpha=0.6):
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
    qtrajs = np.apply_along_axis(lambda x: weighted_quantile(
        x, [1/3, 1/2, 2/3], sample_weight=w), 0, trajs)
    if plot_means:
        # approximate quantiles as distance from median applied to mean
        # with clips
        mtrajs = np.sum(trajs * w[:, np.newaxis, np.newaxis], axis=0)
        qtrajs[0, :, :] = np.clip(
            qtrajs[0, :, :] - qtrajs[1, :, :] + mtrajs, 0, 1)
        qtrajs[2, :, :] = np.clip(
            qtrajs[2, :, :] - qtrajs[1, :, :] + mtrajs, 0, 1)
        qtrajs[1, :, :] = mtrajs
    if ax is None:
        ax = plt.gca()
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Fraction of Population')
    for i in range(trajs.shape[-1]):
        ax.plot(x, qtrajs[1, :, i],
                color=f'C{i}', label=f'Compartment {names[i]}')
        ax.fill_between(x, qtrajs[0, :, i], qtrajs[-1, :, i],
                        color=f'C{i}', alpha=alpha)
    if not plot_means:
        ax.plot(x, np.sum(qtrajs[1, :, :], axis=1),
                color='gray', label='Total', linestyle=':')

    if add_legend:
        # add margin for legend
        ax.set_xlim(0, max(x))
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))


def merge_history(base, other, prefix=''):
    if base is None:
        return other
    if other is None:
        return base
    for k, v in other.history.items():
        if prefix + k in other.history:
            base.history[prefix + k].extend(v)
        else:
            base.history[prefix + k] = v
    return base

def exposed_finder(sampled_trajs):
    R'''
    Finds the initial exposed patch (t=0) for trajs
    '''
    if len(sampled_trajs.shape) < 4:
        sampled_trajs = sampled_trajs[np.newaxis, ...]
    exposed_sampled_trajs = sampled_trajs[:, 0, :, 1]
    return np.where(exposed_sampled_trajs > 0)[:][1]


def weighted_exposed_prob_finder(prior_exposed_patch, meta_pop_size, weights=None):
    R'''
    Finds the weighted probabiity of being exposed in every patch at time zero across all the sample trajs
    '''
    if weights is None:
        weights = np.ones_like(prior_exposed_patch)
    posterior_exposed = np.zeros((meta_pop_size))
    for i, m in enumerate(prior_exposed_patch):
        posterior_exposed[m] += weights[i]
    posterior_exposed /= np.sum(posterior_exposed)
    return posterior_exposed


def p0_map(prior_exposed_patch, meta_pop_size, weights=None, patch_names=None, title=None,
           choropleth=False, geojson=None, fontsize=12, figsize=(15, 8)):
    R'''
    Plots the weighted probabiity of being exposed in every patch at time zero on a grid or
    on a choropleth map (this requires geopandas and geoplot packages).
    '''
    weighted_exposed_prob = maxentep.weighted_exposed_prob_finder(
        prior_exposed_patch, meta_pop_size, weights=weights)
    if choropleth:
        import geopandas as gpd
        import geoplot as gplt
        import geoplot.crs as gcrs

        census_geo = gpd.read_file(geojson).sort_values(by=['county']).assign(
            prob_exposed_initial=weighted_exposed_prob)
        ax = gplt.choropleth(
            census_geo,
            hue='prob_exposed_initial',
            cmap='Reds', linewidth=0.5,
            edgecolor='k',
            legend=True,
            projection=gcrs.AlbersEqualArea(),
            figsize=figsize,
        )
        plt.title(title, fontsize=fontsize)
    else:
        nrow = int(np.floor(np.sqrt(meta_pop_size)))
        ncol = int(np.ceil(meta_pop_size / nrow))
        z = weighted_exposed_prob
        for i in range(meta_pop_size % ncol):
            z = np.append(z, 0)
        z = z.reshape(nrow, ncol)
        if patch_names is None:
            patch_names = np.arange(meta_pop_size)
            for i in range(meta_pop_size % ncol):
                patch_names = np.append(patch_names, ' ')
            patch_names = patch_names.reshape(nrow, ncol)
        else:
            for i in range(meta_pop_size % ncol):
                patch_names = np.append(patch_names, ' ')
            # [:-7] is to remove 'County' from string
            patch_names = np.array([m[:-7]
                                    for m in patch_names]).reshape(nrow, ncol)

        fig, ax = plt.subplots(figsize=figsize)
        c = ax.pcolor(z, edgecolors='k', linewidths=0.4,
                      linestyle='-', cmap='Reds')
        plt.gca().set_aspect("equal")
        plt.gcf().set_size_inches(figsize)

        for i in range(nrow):
            for j in range(ncol):
                if i * ncol + j == meta_pop_size:
                    break
                plt.text(j+0.38, i+0.38,
                         patch_names[i, j], color="#2480c7", fontsize=fontsize)
        fig.colorbar(c, ax=ax)
        ax.set_title(title, fontsize=fontsize+10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()


def compartment_restrainer(restrained_patches, restrained_compartments, ref_traj, prior, npoints=5, noise=0, start_time=0, end_time=None, time_average=7):
    R'''
    Adds restraints to reference traj based on selected compartments of selected patches 
    '''
    if tf.rank(ref_traj).numpy() != 4:
        ref_traj = ref_traj[tf.newaxis, ...]
    M = ref_traj.shape[2]
    number_of_restrained_patches = len(restrained_patches)
    number_of_restrained_compartments = len(restrained_compartments)
    if number_of_restrained_patches > M:
        raise ValueError(
            "Oops! Number of patches to be restrained exceeeds the total number of patches.")
    if end_time is None:
        end_time = ref_traj.shape[1]//2
    print(f'Restraints are set in this time range: [{start_time}, {end_time}]')
    # example if number_of_restraint_patches = 2 : (recovered and infected patch)
    restraints = []
    plot_fxns_list = []
    for i in range(number_of_restrained_patches):
        plot_fxns = []
        for j in range(number_of_restrained_compartments):
            res, plfxn = maxentep.traj_to_restraints(ref_traj[0, :, :, :], [
                restrained_patches[i], restrained_compartments[j]], npoints, prior, noise, time_average, start_time=start_time, end_time=end_time)
            restraints += res
            plot_fxns += plfxn
        plot_fxns_list.append(plot_fxns)
    return restraints, plot_fxns_list


def get_dist(prior_prams, compartments=['E', 'A', 'I', 'R']):
    R_dist = []
    T_dist = []
    start_dist = []
    beta_dist = []
    for i in range(len(prior_prams)):
        param_batch = prior_prams[i]
        R_dist.append(param_batch[0])
        T_dist.append(param_batch[1])
        start_dist.append(param_batch[2])
        beta_dist.append(param_batch[3])
    R_dist = tf.concat(R_dist, axis=0)
    T_dist = tf.concat(T_dist, axis=0)
    start_dist = tf.concat(start_dist, axis=0)
    beta_dist = tf.concat(beta_dist, axis=0)
    # get eta
    E_A = 1/T_dist[:, compartments.index('E'), compartments.index('A')].numpy()
    # get alpha
    A_I = 1/T_dist[:, compartments.index('A'), compartments.index('I')].numpy()
    # get mu
    I_R = 1/T_dist[:, compartments.index('I'), compartments.index('R')].numpy()
    # Getting starting exposed fraction
    mask = tf.greater(start_dist, 0)
    start_exposed_dist = tf.boolean_mask(start_dist, mask).numpy()
    return [R_dist, E_A, A_I, I_R, start_exposed_dist, beta_dist]


def plot_dist(R_dist, E_A, A_I, I_R, start_exposed_dist, beta_dist, name='prior'):
    import seaborn as sns
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 6), dpi=200)
    fig.suptitle(f'Parameter {name} distributions', fontsize=20, y=1.00)
    sns.distplot(x=beta_dist, ax=axs[0, 0], axlabel='Beta')
    sns.distplot(x=start_exposed_dist,
                 ax=axs[0, 1], axlabel='Exposed start fraction')
    sns.distplot(x=R_dist, ax=axs[0, 2], axlabel='Mobility matrix')
    sns.distplot(x=E_A, ax=axs[1, 0], axlabel=r'$\eta^{-1}$ : E->A (days)')
    sns.distplot(x=A_I, ax=axs[1, 1], axlabel=r'$\alpha ^{-1}$ : A->I (days)')
    sns.distplot(x=I_R, ax=axs[1, 2], axlabel=r'$\mu^{-1}$ : I->R (days)')


def graph_degree(graph):
    R'''
    Returns degree-of-freedom of a network graph based on networkx graph
    '''
    degree = len(list(graph.edges))/len(list(graph.nodes))
    return degree

def gen_graph(M):
    R'''
    Returns a fully connected dense networkx graph of size M, edge list and node list
    '''
    import networkx as nx
    G = nx.DiGraph()
    edge_list = []
    k = 0
    i = 0
    node_list = range(M)
    for k in range(M):
        G.add_nodes_from([node_list[k]])
        for i in range(M):
            edge_list.append((i, k))
    G.add_edges_from(edge_list)
    return G, edge_list, node_list

def gen_random_graph(M, p=1.0):
    R'''
    Returns a random networkx graph of size M with connection probability p, edge list and node list
    '''
    import networkx as nx
    graph = nx.fast_gnp_random_graph(M, p, directed=True)
    # adding self-connection
    edge_list = [(i, i) for i in range(M)]
    graph.add_edges_from(edge_list)
    return graph


def draw_graph(graph, weights=None, heatmap=False, title=None, dpi=150, true_origin=None):
    R'''
    Plots networkx graph. Heatmap option changes node color based on node weights.
    '''
    import networkx as nx
    if heatmap:
        options = {
            'width': 0.7,
            'edge_color': '#1d4463',
            'font_color': '#827c60',
            'node_size': 500
        }
        if weights is None:
            M = len(graph.nodes)
            weights = np.ones(M)/M
        max_weight = float(max(weights))
        node_colors = [plt.cm.Reds(weight/max_weight) for weight in weights]
        edge_colors = ['k'] * len(weights)
        line_widths = [0.5] * len(weights)
        if true_origin:
            edge_colors[true_origin] = '#e8c900'
            line_widths[true_origin] = 2.5
        colors_unscaled = [tuple(map(lambda x: max_weight*x, y))
                           for y in node_colors]
        # Creating a dummy colormap
        heatmap = plt.pcolor(colors_unscaled, cmap=plt.cm.Reds)
        plt.close()
        fig, ax = plt.subplots(dpi=dpi)
        graph_plt = nx.draw(graph, with_labels=True, pos=nx.shell_layout(graph), font_weight='bold',
                            node_color=node_colors, linewidths=line_widths, ax=ax, **options, cmap='Reds')
        ax = plt.gca()  # to get the current axis
        ax.collections[0].set_edgecolor(edge_colors)
        ax.collections[0].set_linewidths(line_widths)
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel('Patient-zero Probability',
                           labelpad=15, rotation=90)
    else:
        fig, ax = plt.subplots(dpi=dpi)
        options = {
            'node_color': '#0a708c',
            'width': 0.7,
            'edge_color': '#555555',
            'font_color': '#ffffff',
            'node_size': 500,
        }
        nx.draw(graph, with_labels=True, pos=nx.shell_layout(
            graph), font_weight='bold', ax=ax, **options)
    if title:
        plt.title(title, fontsize=20)
    return


def sparse_graph_mobility(sparse_graph, fully_connected_mobility_matrix):
    R'''
    Generates a sprase mobility matrix based on sparse graph and a fully connected mobility matrix inputs.
    For a fully connected graph, the output mobility matrix remains the same.
    '''
    sparse_mobility_matrix = np.zeros_like(fully_connected_mobility_matrix)
    for i, edge in enumerate(sparse_graph.edges()):
        sparse_mobility_matrix[edge[0], edge[1]
                               ] = fully_connected_mobility_matrix[edge[0], edge[1]]
    return sparse_mobility_matrix


def p0_loss(trajs, weights, true_p0_node):
    R'''Returns cross-entropy loss for p0 based on sampled trajs and maxent weights, size of meta-population and ground-truth p0 node inputs'''
    M = trajs.shape[2]
    prior_exposed_patch = maxentep.exposed_finder(trajs)
    weighted_exposed_prob = maxentep.weighted_exposed_prob_finder(
        prior_exposed_patch, M, weights=weights)
    loss = -np.log(weighted_exposed_prob[true_p0_node])
    return loss


def traj_loss(ref_traj, trajs, weights):
    R'''Returns KL divergence loss for predicted traj based on ref_traj, sampled trajs and maxent weights inputs'''
    M = trajs.shape[2]
    Time = trajs.shape[1]
    weights /= tf.reduce_sum(weights)
    mtrajs_counties = tf.reduce_sum(
        trajs * weights[:, tf.newaxis, tf.newaxis, tf.newaxis], axis=0)
    loss = -tf.reduce_sum(ref_traj*tf.math.log(
        tf.math.divide_no_nan(mtrajs_counties, ref_traj) + 1e-10))/M/Time
    return loss
