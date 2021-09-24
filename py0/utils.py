import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import py0
import tensorflow as tf
import maxent


class TransitionMatrix:
    ''' Defines the transition between different compartments in the disease model, given different epidemiology parameters.
    '''
    def __init__(self, compartment_names, infectious_compartments):
        ''' 
        :param compartment_names: name of different compartments
        :type compartment_names: list of compartment_names as strings
        :param infectious_compartments: index of infectious compartment_names
        :type infectious_compartments: list
        '''
        self.names = compartment_names
        self.infectious_compartments = infectious_compartments
        self.transitions = []
        self.mat = None

    def add_transition(self, name1, name2, time, time_var):
        '''
        :param name1: source compartment
        :type name1: string
        :param name2: destination compartment
        :type name2: string
        :param time: time it takes to move from source compartment to destination compartment. This is typically the reciprocal of rates.
        :type time: float
        :param time_var: variance of the time it takes to move from source compartment to destination compartment. Use zero unless you are
            creating an esemble of trajectories to do inference using MaxEnt.
        :type time_var: float
        '''
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
        ''' Returns matrix value.
        '''
        if self.mat is None:
            self._make_matrix()
        return self.mat


def weighted_quantile(values, quantiles, sample_weight=None,
                       values_sorted=False, old_style=False):
    ''' Very close to numpy.percentile, but supports weights.
        Note: quantiles should be in [0, 1]!

    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.

    :return: numpy.array with computed quantiles.
    '''
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


def patch_quantile(trajs, *args, ref_traj=None, weights=None, lower_q_bound=1/3, upper_q_bound=2/3, restrained_patches=None, plot_fxns_list=None, figsize=(18, 18), patch_names=None,
                   fancy_shading=False, n_shading_gradients=30, alpha=0.6, obs_color='C0', yscale_log=False, ** kw_args):
    ''' Does ``traj_quantile`` for trajectories of shape [N, T, M, C] where N is the number of samples, T is the number of timesteps,
        M is the number of patches (nodes) and C is the number of compartments.

    :param trajs: ensemble of trajectories after sampling
    :type trajs: tensor with dtype tf.float32 of shape [N, T, M, C] where N is the number of samples, T is the number of timesteps,
        M is the number of patches (nodes) and C is the number of compartments.
    :param ref_traj: reference trajectory
    :type ref_traj: tensor with dtype tf.float32 of shape [1, T, M, C] where T is the number of timesteps,
        M is the number of patches (nodes) and C is the number of compartments.
    :param weights: weights for the each trajectory in the ensemble. If not defined uniform weights will be assumed.
    :type weights: tensor with dtype tf.float32
    :param lower_q_bound: lower quantile bound
    :type lower_q_bound: float
    :param upper_q_bound: upper quantile bound
    :type upper_q_bound: float
    :param restrained_patches: index of the patches (nodes) restrained
    :type restrained_patches: list
    :param plot_fxns_list: output of `compartment_restrainer`
    :type plot_fxns_list: list
    :param figsize: figure size
    :type figsize: tupple
    :param patch_names: name of the patches (nodes). If not provided patches name will be defined by their index.
    :type patch_names: list
    :param fancy_shading: allows for gradient shading of the confidence interval
    :type fancy_shading: bool
    :param n_shading_gradients: number of intervals for shading gradients
    :type n_shading_gradients: int
    :param alpha: alpha value for edges that allows transparency
    :type alpha: float
    :param obs_color: marker color for the observation nodes
    :type obs_color: string
    :param yscale_log: change y axis scale to log for better visualization of the observations.
    :type yscale_log: bool
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
            if ref_traj is not None:
                ax[i, j].plot(ref_traj[0, :, i * ncol + j, :], linestyle='--')
            traj_quantile(trajs[:, :, i * ncol + j, :], *args, ax=ax[i, j], weights=weights, lower_q_bound=lower_q_bound, upper_q_bound=upper_q_bound,
                          add_legend=i == 0 and j == ncol - 1, fancy_shading=fancy_shading, n_shading_gradients=n_shading_gradients, alpha=alpha, **kw_args)
            ax[i, j].set_ylim(0, 1)
            if restrained_patches is not None and i * ncol + j in restrained_patches:
                for _,pf in enumerate(plot_fxns_list[restrained_patches.tolist().index(i * ncol + j)]):
                    pf(ax[i, j], 0, color=obs_color)
                ax[i, j].spines['bottom'].set_color(obs_color)
                ax[i, j].spines['top'].set_color(obs_color)
                ax[i, j].spines['right'].set_color(obs_color)
                ax[i, j].spines['left'].set_color(obs_color)
                ax[i,j].spines['left'].set_linewidth(2)
                ax[i,j].spines['top'].set_linewidth(2)
                ax[i,j].spines['right'].set_linewidth(2)
                ax[i,j].spines['bottom'].set_linewidth(2)
            if patch_names is None:
                ax[i, j].text(trajs.shape[1] // 2, 0.8,
                              f'Patch {i * ncol + j}')
            else:
                patch_names = patch_names
                ax[i, j].set_title(patch_names[i * ncol + j])

            if j == 0 and i == nrow // 2:
                ax[i, j].set_ylabel('Population Fraction')
            if i == nrow - 1:
                ax[i, j].set_xlabel('Time (days)')
            if yscale_log:
                ax[i, j].set_yscale('log')
                ax[i, j].set_ylim([1e-4, 1])
            if j >= NP % ncol:
                ax[nrow-1, j].set_visible(False)

    plt.tight_layout()


def traj_quantile(trajs, weights=None, lower_q_bound=1/3, upper_q_bound=2/3,  figsize=(9, 9), names=None, plot_means=True, ax=None,
                  add_legend=True, alpha=0.6, fancy_shading=False, n_shading_gradients=30):
    ''' Make a plot of all the trajectories and the average trajectory based on trajectory weights and lower and upper quantile values.
    
    :param trajs: ensemble of trajectories after sampling
    :type trajs: tensor with dtype tf.float32 of shape [N, T, M, C] where N is the number of samples, T is the number of timesteps,
        M is the number of patches (nodes) and C is the number of compartments.
    :param weights: weights for the each trajectory in the ensemble. If not defined uniform weights will be assumed.
    :type weights: tensor with dtype tf.float32
    :param lower_q_bound: lower quantile bound
    :type lower_q_bound: float
    :param upper_q_bound: upper quantile bound
    :type upper_q_bound: float
    :param figsize: figure size
    :type figsize: tupple
    :param names: name of compartments as strings
    :type names: list
    :param plot_means: if ``True`` approximates quantiles as distance from median applied to mean.
    :type plot_means: bool
    :param ax: ``matplotlib.axes.AxesSubplot``. Defaults to a new axis.
    :param add_legend: show legend 
    :type add_legend: bool
    :param alpha: alpha value for edges that allows transparency
    :type alpha: float
    :param fancy_shading: allows for gradient shading of the confidence interval
    :type fancy_shading: bool
    :param n_shading_gradients: number of intervals for shading gradients
    :type n_shading_gradients: int
    '''

    if lower_q_bound+upper_q_bound != 1.0:
        raise ValueError('lower and upper quantile bounds should sum up to 1.0.')
    if names is None:
        names = [f'Compartment {i}' for i in range(trajs.shape[-1])]
    if weights is None:
        w = np.ones(trajs.shape[0])
    else:
        w = weights
    w /= np.sum(w)
    x = range(trajs.shape[1])
    fancy_lower_q_bounds = np.linspace(lower_q_bound, 0.5, n_shading_gradients)
    fancy_higher_q_bounds = 1 - fancy_lower_q_bounds
    for n in range(n_shading_gradients):
        if not fancy_shading:
            lower_q_bound = fancy_lower_q_bounds[0]
            upper_q_bound = fancy_higher_q_bounds[0]
        else:
            lower_q_bound = fancy_lower_q_bounds[n]
            upper_q_bound = fancy_higher_q_bounds[n]
        # weighted quantiles doesn't support axis
        # fake it using apply_along
        qtrajs = np.apply_along_axis(lambda x: weighted_quantile(
            x, [lower_q_bound, 1/2, upper_q_bound], sample_weight=w), 0, trajs)
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
            ax.fill_between(x, qtrajs[0, :, i], qtrajs[-1, :, i],
                            color=f'C{i}', alpha=alpha, linewidth=0.0, rasterized=True)
            if n == 0:
                ax.plot(x, qtrajs[1, :, i],
                        color=f'C{i}', label=f'Compartment {names[i]}')
                ax.plot(x, qtrajs[0, :, i],
                        color=f'C{i}', alpha=0.4, linewidth=1)
                ax.plot(x, qtrajs[2, :, i],
                        color=f'C{i}', alpha=0.4, linewidth=1)
        if not fancy_shading:
            break
    if not plot_means:
        ax.plot(x, np.sum(qtrajs[1, :, :], axis=1),
                color='gray', label='Total', linestyle=':')

    if add_legend:
        # add margin for legend
        ax.set_xlim(0, max(x))
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        # removing duplicates from legend
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(
            zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.05, 1))


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


def exposed_finder(trajs):
    ''' Finds the initial exposed patch (t=0) for trajs

    :param trajs: ensemble of trajectories after sampling
    :type trajs: tensor with dtype tf.float32 of shape [N, T, M, C] where N is the number of samples, T is the number of timesteps,
        M is the number of patches (nodes) and C is the number of compartments.

    :return: A numpy array containing the index of the initial exposed node for the ensemble of trajectories
    '''
    if len(trajs.shape) < 4:
        trajs = trajs[np.newaxis, ...]
    exposed_sampled_trajs = trajs[:, 0, :, 1]
    return np.where(exposed_sampled_trajs > 0)[:][1]


def weighted_exposed_prob_finder(prior_exposed_patch, meta_pop_size, weights=None):
    ''' Finds the weighted probability of being exposed in every patch at time zero across all the sample trajs.

    :param prior_exposed_patch: output of ``exposed_finder`` function.
    :type prior_exposed_patch: an array of size N (sample size)
    :param meta_pop_size: size of the metapopulation
    :type meta_pop_size: int
    :param weights: weights of the trajectories in the ensemble. If not provided, will be assumed as 1/N.
    :type weights: tensor with dtype tf.float32

    :return: weighted probability of being exposed across all patches
    '''
    if weights is None:
        weights = np.ones_like(prior_exposed_patch)
    posterior_exposed = np.zeros((meta_pop_size))
    for i, m in enumerate(prior_exposed_patch):
        posterior_exposed[m] += weights[i]
    posterior_exposed /= np.sum(posterior_exposed)
    return posterior_exposed


def p0_map(prior_exposed_patch, meta_pop_size, weights=None, patch_names=None, title=None,
           choropleth=False, geojson=None, fontsize=12, figsize=(15, 8), vmin=None, vmax=None,
           restrained_patches=None, true_origin=None, obs_size=5, obs_color='C0', org_color='C8',
           colormap='Reds', ax=None, projection=None, show_legend=True, show_cbar=True):
    ''' Plots the weighted probabiity of being exposed in every patch at time zero on a grid or
    on a choropleth map (this requires geopandas and geoplot packages). If choropleth plotting is enabled, make sure your geojson has 'county' as header for 
    the counties name column and your patches names are alphabetically sorted.

    :param prior_exposed_patch: output of ``exposed_finder`` function.
    :type prior_exposed_patch: an array of size N (sample size)
    :param meta_pop_size: size of the metapopulation
    :type meta_pop_size: int
    :param weights: weights of the trajectories in the ensemble. If not provided, will be assumed as 1/N.
    :type weights: tensor with dtype tf.float32
    :param patch_names: name of the patches. Note that the namings should be similar to names from GeoJSON if using ``choropleth``.
    :type patch_names: list
    :param title: figure title
    :type title: string
    :param choropleth: turn on if plotting choropleth plots
    :type choropleth: bool
    :param geojson: path to GeoJSON file describing the geographic features of the metapopulation
    :type geojson: string
    :param fontsize: font size
    :type fontsize: float
    :param figsize: figure size
    :type figsize: tupple
    :param vmin: minimum value of the color bar
    :type vmin: float
    :param vmax: maximum value of the color bar
    :type vmax: float
    :param restrained_patches: index of the patches (nodes) restrained
    :type restrained_patches: list
    :param true_origin: index for the true origin node
    :type true_origin: int
    :param obs_size: marker size for the observation nodes
    :type obs_size: float
    :param obs_color: marker color for the observation nodes
    :type obs_color: string
    :param org_color: marker size for the true origin node
    :type org_color: tensor with dtype tf.float32
    :param colormap: Matplotlib colormaps 
    :type colormap: string
    :param ax: ``matplotlib.axes.AxesSubplot``. Defaults to a new axis.
    :param projection: the projection to use. For reference see
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    :param show_legend: show legend for true origin or obsevations. 
    :type show_legend: bool
    :param show_cbar: show heatmap color bar 
    :type show_cbar: bool
    '''
    weighted_exposed_prob = py0.weighted_exposed_prob_finder(
        prior_exposed_patch, meta_pop_size, weights=weights)
    if choropleth:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                'This function requires geopandas package to run. Please install the missing dependency.')
        try:
            import geoplot as gplt
            import geoplot.crs as gcrs
        except ImportError:
            raise ImportError(
                'This function requires geoplot package to run. Please install the missing dependency.')
        import matplotlib as mpl
        import matplotlib.lines as mlines
        if vmax is None:
            vmax = max(weighted_exposed_prob)
        if vmin is None:
            vmin = min(weighted_exposed_prob)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=colormap).cmap
        census_geo = gpd.read_file(geojson).sort_values(by=['county']).assign(
            prob_exposed_initial=weighted_exposed_prob)
        total_bounds = census_geo.total_bounds + np.array([0, -0.2, 0, 0])
        if projection is None:
            projection = gcrs.AlbersEqualArea(
                central_longitude=-121, central_latitude=39.5)
        if ax is not None:
            gplt.choropleth(
                census_geo,
                hue='prob_exposed_initial',
                cmap=cmap, norm=norm, linewidth=0.5,
                edgecolor='k',
                legend=False,
                extent=total_bounds,
                zorder=0,
                ax=ax,
                figsize=figsize)
        else:
            ax = gplt.choropleth(
                census_geo,
                hue='prob_exposed_initial',
                cmap=cmap, norm=norm, linewidth=0.5,
                edgecolor='k',
                legend=False,
                projection=projection,
                figsize=figsize,
                extent=total_bounds,
                zorder=0)
        ax.set_facecolor('w')
        # scatter plot the observations on the map
        handles = []
        census_geo_points = census_geo.copy()
        census_geo_points['geometry'] = census_geo_points['geometry'].centroid
        if restrained_patches is not None:
            restrained_patches_names = [' '.join(patch_names[i].split()[:-1])
                                        for i in restrained_patches]
            obs = census_geo_points.query(
                'county == @restrained_patches_names')
            gplt.pointplot(obs, ax=ax,
                           marker='o', s=obs_size, color=obs_color, zorder=2, extent=total_bounds, edgecolor='#ebebeb', alpha=0.9)
            obs_marker = mlines.Line2D([], [], color=obs_color, marker='o', linestyle='None',
                                       markersize=obs_size, label='Observation(s)', markeredgecolor='#ebebeb')
            handles.append(obs_marker)
        if true_origin is not None:
            origin_name = patch_names[true_origin]
            origin_name = ' '.join(origin_name.split()[:-1])
            org = census_geo_points.query('county == @origin_name')
            gplt.pointplot(org, ax=ax,
                           marker='v', s=obs_size+6, color=org_color, zorder=1, extent=total_bounds, edgecolor='#ebebeb', alpha=0.9)
            org_marker = mlines.Line2D([], [], color=org_color, marker='v', linestyle='None',
                                       markersize=obs_size+6, label='True Origin', markeredgecolor='#ebebeb')
            handles.append(org_marker)
        if show_legend:
            pos_ax = ax.get_position()
            ax.legend(handles=handles, bbox_to_anchor=[-0.05*(pos_ax.width), pos_ax.height],
                      frameon=True, fontsize=fontsize, edgecolor='k', facecolor='#ebebeb')

        if show_cbar:
            cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax,
                                      pad=0.05, fraction=0.03)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.set_title('$P_0$ Probability', fontsize=fontsize, y=1.05)
        if title:
            ax.set_title(title, fontsize=fontsize+10, y=1.05)
        plt.tight_layout()
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


def compartment_restrainer(restrained_patches, restrained_compartments, ref_traj, prior, npoints=5, noise=0,
    start_time=0, end_time=None, time_average=7, marker_size=10, marker_color='r'):
    ''' Adds restraints to reference traj based on selected compartments of selected patches.

    :param restrained_patches: index of the patches (nodes) restrained
    :type restrained_patches: list
    :param restrained_compartments: index values for the restrained compartments
    :type restrained_compartments: list
    :param ref_traj: reference traj
    :type ref_traj:  a [1, T, M, C] tensor with dtype tf.float32, where T is the number of timesteps, M is the number of patches (nodes) and
        C is the number of compartments.
    :param prior: Prior distribution for expected deviation from target for restraint. Can be either 'EmptyPrior' for exact agreement
        or set to 'Laplace' for more allowable disagreement.
    :type prior: maxent.prior
    :param npoints: number of data points in each restrained compartment
    :type npoints: int
    :param noise: multiplicative noise to be added to observations to allow higher uncertainty
    :type noise: float
    :param start_time: index for the lower time limit of restraints
    :type start_time: int
    :param end_time: index for the higher time limit of restraints. If not provided, maximum timestep will be assumed.
    :type end_time: int
    :param time_average: number of timesteps to for time averaging of restraints
    :type time_average: int
    :param marker_size: marker size for restraints
    :type marker_size: int
    :param marker_color: marker color for restraints
    :type marker_color: string

    :return: restraints, plot_fxns_list
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
            res, plfxn = py0.traj_to_restraints(ref_traj[0, :, :, :], [
                restrained_patches[i], restrained_compartments[j]], npoints, prior, noise, time_average, start_time=start_time,
                end_time=end_time, marker_size=marker_size, marker_color=marker_color)
            restraints += res
            plot_fxns += plfxn
        plot_fxns_list.append(plot_fxns)
    return restraints, plot_fxns_list


def get_dist(prior_params, compartments=['E', 'A', 'I', 'R']):
    ''' Gets distributions for the model parameters in the ensemble trectory sampling.

    :param prior_params: model parameters during sampling over different batches.
    :type prior_params: list
    :param compartments: list of compartments except for 'S' (susceptible) as strings.
    :type compartments: list

    return: list of model parameter's distributions.
    '''
    R_dist = []
    T_dist = []
    start_dist = []
    beta_dist = []
    for i in range(len(prior_params)):
        param_batch = prior_params[i]
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
    ''' plots a ``seaborn.distplot`` for the model's prior parameter distribution.

    :param R_dist:  sampled mobility flows as tensor with dtype tf.float32
    :param E_A: time for going from E->A as tensor with dtype tf.float32
    :param A_I: time for going from A->I as tensor with dtype tf.float32
    :param I_R: time for going from I->R as tensor with dtype tf.float32
    :param start_exposed_dist: starting exposed fraction as tensor with dtype tf.float32
    :param beta_dist: beta value(s) as tensor with dtype tf.float32
    :param name: name for the distributions that shows up in the figure title
    :type name: string

    '''
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
    ''' Returns graph degree of a network graph based on networkx graph input.
    
    :param graph:  networkx graph

    :return: graph degree
    '''
    degree = len(list(graph.edges))/len(list(graph.nodes))
    return degree

def gen_graph(M):
    ''' Generates a fully connected dense networkx graph of size M, edge list and node list.

    :param M: number of nodes in the metapopulation
    :type M: int
    
    :return: graph, edge list, node list
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


def gen_graph_from_R(mobility_matrix):
    ''' Generates a networkx graph of size mobility_matrix.shape[0], edge list and node list.

    :param mobility_matrix: mobility flows between the nodes
    :type mobility_matrix: numpy array of [M, M], where M is the number of nodes in the metapopulation
    
    :return: graph, edge list, node list
    '''
    import networkx as nx
    M = mobility_matrix.shape[0]
    G = nx.DiGraph()
    edge_list = []
    k = 0
    i = 0
    node_list = range(M)
    for k in range(M):
        G.add_nodes_from([node_list[k]])
        for i in range(M):
            if mobility_matrix[k, i] != 0:
                edge_list.append(
                    (k, i, {'weight': np.log(mobility_matrix[k, i])}))
    G.add_edges_from(edge_list)
    return G, edge_list, node_list


def gen_random_graph(M, p=1.0, seed=None):
    ''' Returns a random networkx graph of size M with connection probability p

    :param M: number of nodes in the metapopulation
    :type M: int
    :param p: node connection probability
    :type p: float
    :param seed: allows random seeding for graph generations

    :return: graph
    '''
    import networkx as nx
    graph = nx.fast_gnp_random_graph(M, p, directed=True, seed=seed)
    # adding self-connection
    edge_list = [(i, i) for i in range(M)]
    graph.add_edges_from(edge_list)
    return graph


def draw_graph(graph, weights=None, heatmap=False, title=None, dpi=150, true_origin=None, color_bar=True):
    ''' Plots networkx graph.

    :param graph:  networkx graph
    :param weights:  probabiity of being exposed in every patch at time zero across all the sample trajs. If not provided uniform
        probability will be assumed over all nodes.
    :param heatmap: change node color based on weights
    :type heatmap: bool
    :param title: plot title 
    :type title: string
    :param dpi: dpi value of plot
    :type dpi: int
    :param true_origin: index for the true origin node
    :type true_origin: int
    :param color_bar: enables color bar in plot
    :type color_bar: bool
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
        if color_bar:
            cbar = plt.colorbar(heatmap)
            cbar.ax.set_ylabel('$P_0$ Probability',
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


def draw_graph_on_map(mobility_matrix, geojson, title=None, node_size=300, node_color="#eb4034", edge_color='#555555',
                      map_face_colors=None, fontsize=12, figsize=(10, 10), ax=None, alpha_edge=0.3, alpha_node=0.8,
                      true_origin=None, restrained_patches=None, obs_color='C0', org_color='C8', show_map_only=False, show_legend=False):
    ''' Plots networkx graph on a map given mobility flows and geographic features. The edges are weighted by the mobility flows between the nodes.
    This function also alows for visualization of true orgin node and observed ones.

    :param mobility_matrix: mobility flows between the nodes
    :type mobility_matrix: numpy array of [M, M], where M is the number of nodes in the metapopulation
    :param geojson: path to GeoJSON file describing the geographic features of the metapopulation
    :type geojson: string
    :param title: figure title
    :type title: string
    :param node_size: size for the networkx nodes
    :type node_size: int
    :param node_color: color for the networkx nodes
    :type node_color: string or a list of strings with length M
    :param edge_color: color for the networkx edges
    :type edge_color: string
    :param map_face_colors: color for the faces of patches on the map
    :type map_face_colors: string or a list of strings with length M
    :param fontsize: font size
    :type fontsize: float
    :param figsize: figure size
    :type figsize: tupple
    :param ax: ``matplotlib.axes.AxesSubplot``. Defaults to a new axis.
    :param show_legend: show legend for true origin or obsevations.
    :param alpha_edge: alpha value for edges that allows transparency
    :type alpha_edge: float
    :param alpha_nodes: alpha value for nodes that allows transparency
    :type alpha_nodes: float
    :param true_origin: index for the true origin node
    :type true_origin: int
    :param restrained_patches: index of the patches (nodes) restrained
    :type restrained_patches: list
    :param obs_color: marker color for the observation nodes
    :type obs_color: string
    :param org_color: marker size for the true origin node
    :type org_color: tensor with dtype tf.float32
    :param show_map_only: Showing the map only without the networkx graph. Default is ``False``.
    :type show_map_only: bool
    :type show_legend: bool
    '''
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            'This function requires geopandas package to run. Please install the missing dependency.')
    import matplotlib.lines as mlines
    import networkx as nx
    import random
    graph, edge_list, node_list = gen_graph_from_R(mobility_matrix)
    M = len(node_list)
    if map_face_colors is None:
        random.seed(2)
        map_face_colors = ['C'+f'{i}' for i in range(M)]
        random.shuffle(map_face_colors)
        random.shuffle(map_face_colors)
    handles = []
    if true_origin is not None or restrained_patches is not None:
        map_face_colors = ['#c2c2c2']*M
        if true_origin is not None:
            node_color_origin = [node_color]*M
            map_face_colors[true_origin] = org_color
            node_color_origin[true_origin] = org_color
            org_marker = mlines.Line2D([], [], color=org_color, marker='o', linestyle='None',
                                       markersize=node_size/30+5, label='True Origin', markeredgecolor='k')
            handles.append(org_marker)
            node_size_origin = [node_size]*M
            node_size_origin[true_origin] *= 2.5
        if restrained_patches is not None:
            node_color_obs = [node_color]*M
            edge_color_obs = ['k']*M
            node_size_obs = [node_size]*M
            if true_origin is not None:
                node_color_obs[true_origin] = org_color
                edge_color_obs[true_origin] = 'none'
            for i in restrained_patches:
                map_face_colors[i] = obs_color
                node_color_obs[i] = obs_color
            obs_marker = mlines.Line2D([], [], color=obs_color, marker='o', linestyle='None',
                                       markersize=node_size/30, label='Observation(s)', markeredgecolor='k')
            handles.append(obs_marker)
            node_color = node_color_obs
    regions = gpd.read_file(geojson).sort_values(by=['county'])
    centroids = np.column_stack((regions.centroid.x, regions.centroid.y))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    regions.plot(linewidth=1, edgecolor='k',
                 facecolor=map_face_colors, alpha=0.3, ax=ax)
    ax.axis("off")
    if not show_map_only:
        positions = dict(zip(graph.nodes, centroids))
        graph_nodes_only = py0.gen_random_graph(M, p=0)
        edge_weights = nx.get_edge_attributes(graph, 'weight').values()
        if not bool(edge_weights):
            print('No weights found for edges. Assuming uniform widths for edges.')
            width = [1]*M
        else:
            width = list(edge_weights)
        if true_origin is None and restrained_patches is None:
            nx.draw(graph, positions, ax=ax, width=width, node_size=node_size, node_color=node_color,
                    edge_color=edge_color, alpha=alpha_edge, edgecolors='k')
            nx.draw(graph_nodes_only, positions, ax=ax,
                    node_color="#eb4034", alpha=alpha_node, edgecolors='k')
        else:
            # draw transparent nodes to get the edges first
            nx.draw(graph, positions, ax=ax, width=width, node_size=node_size, node_color='none',
                    edge_color=edge_color, alpha=alpha_edge, edgecolors='none')
            if true_origin is not None:
                nx.draw(graph_nodes_only, positions, ax=ax, node_color=node_color_origin,
                        node_size=node_size_origin, alpha=1, edgecolors='k')
            if restrained_patches is not None:
                nx.draw(graph_nodes_only, positions, ax=ax, node_color=node_color_obs, node_size=node_size, alpha=alpha_node,
                        edgecolors=edge_color_obs)
    if title:
        ax.set_title(title, fontsize=fontsize, y=1.05)
    if show_legend:
        pos_ax = ax.get_position()
        ax.legend(handles=handles, bbox_to_anchor=[-0.05*(pos_ax.width), pos_ax.height],
                  frameon=True, fontsize=fontsize, edgecolor='k', facecolor='#ebebeb')


def sparse_graph_mobility(sparse_graph, fully_connected_mobility_matrix):
    ''' Generates a sprase mobility matrix based on sparse graph and a fully connected mobility matrix inputs.
    For a fully connected graph, the output mobility matrix remains the same.

    :param graph:  networkx sparse graph
    :param fully_connected_mobility_matrix: [M, M] array with values defining mobility flows between nodes
    :type fully_connected_mobility_matrix: numpy array

    :return: sparsed mobility matrix 
    '''
    sparse_mobility_matrix = np.zeros_like(fully_connected_mobility_matrix)
    for i, edge in enumerate(sparse_graph.edges()):
        sparse_mobility_matrix[edge[0], edge[1]
                               ] = fully_connected_mobility_matrix[edge[0], edge[1]]
    return sparse_mobility_matrix


def p0_loss(trajs, weights, true_origin):
    ''' Returns cross-entropy loss for p0 based on sampled trajs and maxent weights, size of meta-population and ground-truth p0 node inputs.

    :param trajs: sampled ensemble of trajectories
    :type trajs:  a [N, T, M, C] tensor with dtype tf.float32, where N is the number of samples, T is the number of timesteps,
        M is the number of patches (nodes) and C is the number of compartments
    :param weights: weights of the trajectories in the ensemble. If not provided, will be assumed as 1/N.
    :type weights: tensor with dtype tf.float32
    :param true_origin: index for the true origin node
    :type true_origin: int

    :return: cross-entropy loss
    '''
    M = trajs.shape[2]
    prior_exposed_patch = py0.exposed_finder(trajs)
    weighted_exposed_prob = py0.weighted_exposed_prob_finder(
        prior_exposed_patch, M, weights=weights)
    loss = -np.log(weighted_exposed_prob[true_origin])
    return loss


def traj_loss(ref_traj, trajs, weights):
    ''' Returns Kullback–Leibler (KL) divergence loss for predicted traj based on a reference traj and MaxEnt reweighted trajs.

    :param ref_traj: reference traj
    :type ref_traj:  a [1, T, M, C] tensor with dtype tf.float32, where T is the number of timesteps, M is the number of patches (nodes) and
        C is the number of compartments.
    :param trajs: sampled ensemble of trajectories
    :type trajs:  a [N, T, M, C] tensor with dtype tf.float32, where N is the number of samples, T is the number of timesteps,
        M is the number of patches (nodes) and C is the number of compartments

    :return: KL divergence as a scalar value
    '''
    M = trajs.shape[2]
    Time = trajs.shape[1]
    weights /= tf.reduce_sum(weights)
    mtrajs_counties = tf.reduce_sum(
        trajs * weights[:, tf.newaxis, tf.newaxis, tf.newaxis], axis=0)
    loss = -tf.reduce_sum(ref_traj*tf.math.log(
        tf.math.divide_no_nan(mtrajs_counties, ref_traj) + 1e-10))/M/Time
    return loss


def traj_to_restraints(ref_traj, inner_slice, npoints, prior, noise=0.1, time_average=7, start_time=0, end_time=None, marker_size=10, marker_color='r'):
    ''' Creates npoints restraints based on given trajectory with multiplicative noise and time averaging.
    For example, it could be weekly averages with some noise.

    :param ref_traj: reference traj
    :type ref_traj:  a [1, T, M, C] tensor with dtype tf.float32, where T is the number of timesteps, M is the number of patches (nodes) and
        C is the number of compartments.
    :param inner_slice: list of length 2. First index determines the patch and second index determines what compartment on that patch is restrained.
    :type inner_slice: list
    :param npoints: number of data points in each restrained compartment
    :type npoints: int
    :param prior: Prior distribution for expected deviation from target for restraint. Can be either 'EmptyPrior' for exact agreement
        or set to 'Laplace' for more allowable disagreement.
    :type prior: maxent.prior
    :param noise: multiplicative noise to be added to observations to allow higher uncertainty
    :type noise: float
    :param time_average: number of timesteps to for time averaging of restraints
    :type time_average: int
    :param start_time: index for the lower time limit of restraints
    :type start_time: int
    :param end_time: index for the higher time limit of restraints. If not provided, maximum timestep will be assumed.
    :type end_time: int
    :param marker_size: marker size for restraints
    :type marker_size: int
    :param marker_color: marker color for restraints
    :type marker_color: string

    :return: list of restraints, list of functions which take a matplotlib axis and lambda value and plot the restraint on it.
    '''
    if end_time is None:
        end_time = len(ref_traj)
    restraints = []
    plots = []
    # make sure it's a tuple
    inner_slice = tuple(inner_slice)
    try:
        slices = np.random.choice(
            range(start_time // time_average, end_time // time_average), replace=False, size=npoints)
    except ValueError:
        print(f'Only {len(ref_traj) // time_average - start_time // time_average} points are possible given the input time_average = {time_average}.')
    for i in slices:
        # pick random time period
        s = slice(i * time_average,
                  i * time_average + time_average)
        v = np.log(np.clip(np.mean(ref_traj[s], axis=0)[
                    inner_slice] * np.random.normal(loc=1.0, scale=noise), 0, 1) + 1e-15)
        def fxn(x, s=s, j=inner_slice):
            return tf.math.log(tf.reduce_mean(x[s], axis=0)[j] + 1e-15)
        print(i * time_average + time_average // 2 ,
              np.mean(ref_traj[s], axis=0)[inner_slice], np.exp(v))
        # need to make a multiline lambda, so fake it with tuple
        def plotter(ax, l, i=i, v=v, color=marker_color, inner_slice=inner_slice, prior=prior): return (
            ax.plot(i * time_average + time_average // 2 ,
                    np.exp(v), 'o', color=color, markersize=marker_size, markeredgecolor='k'),
            ax.errorbar(i * time_average + time_average // 2 , np.exp(v), xerr=time_average //
                        2, yerr=prior.expected(float(l)), color=color, capsize=3, ms=20)
        )
        r = maxent.Restraint(fxn, v, prior)
        restraints.append(r)
        plots.append(plotter)
    return restraints, plots
