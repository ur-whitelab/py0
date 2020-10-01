from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyabc
from pyabc import ABCSMC
from sbi_gravitation import GravitySimulator, get_observation_points, TRAJECTORY_MAGNITUDE_ADJUSTMENT_FACTOR, prior_means
from abc_gravitation import db_path, distance, model, prior

import seaborn as sns
sns.set_context('paper')
sns.set_style('white',  {'xtick.bottom':True, 'ytick.left':True, 'xtick.color': '#333333', 'ytick.color': '#333333'})
#plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

true_params = [100., 50., 75., 15., -40.]

true_path = np.genfromtxt('true_trajectory.txt')
noisy_path = np.genfromtxt('noisy_trajectory.txt')
sbi_data = np.genfromtxt('samples.txt')

maxent_paths = np.load('maxent_raw_trajectories.npy')
maxent_weights = np.genfromtxt('maxent_traj_weights.txt')

plt.figure()
plt.plot(maxent_weights)
plt.savefig('maxent_weights.png')
plt.close()

maxent_weighted_average_path = np.genfromtxt('maxent_avg_traj.txt')#np.sum(maxent_paths * maxent_weights[:, np.newaxis, np.newaxis], axis=0)#np.average(maxent_paths, axis=0, weights=maxent_weights)
# maxent_weighted_average_params = np.genfromtxt('maxent_weighted_average_params.txt')
# sim = GravitySimulator(maxent_weighted_average_params[0], maxent_weighted_average_params[1], maxent_weighted_average_params[2], [maxent_weighted_average_params[3], maxent_weighted_average_params[4]])
# maxent_weighted_average_path = sim.run()
print(f'maxent weighted average path shape: {maxent_weighted_average_path.shape}')

x_min, y_min = 1000000., 100000.
x_max, y_max = -1000000., -100000.

def get_extrema(path, extrema_list):
    x_min, x_max, y_min, y_max = extrema_list[0], extrema_list[1], extrema_list[2], extrema_list[3]
    x_min = min(x_min, np.min(path[:,0]))
    x_max = max(x_max, np.max(path[:,0]))
    y_min = min(y_min, np.min(path[:,1]))
    y_max = max(y_max, np.max(path[:,1]))
    return [x_min, x_max, y_min, y_max]

extrema = [x_min, x_max, y_min, y_max]

extrema = get_extrema(true_path, extrema)

extrema = get_extrema(true_path, extrema)

sbi_paths = np.zeros([sbi_data.shape[0], noisy_path.shape[0], noisy_path.shape[1]])

observed_points = get_observation_points(noisy_path)

sim = GravitySimulator(prior_means[0], prior_means[1], prior_means[2], prior_means[3:])
prior_means_path = sim.run()

extrema = get_extrema(prior_means_path, extrema)

abc_continued = ABCSMC(model, prior, distance)
abc_continued.load(db_path)
history = abc_continued.history
df, abc_weights = history.get_distribution(m=0, t=history.max_t)
print(abc_weights)
param_means = []
#TODO: compute the cross-entropy of prior vs posterior for all methods -> P*ln(Q), P is prior, Q is posterior.
abc_trajs = np.zeros((len(df), 100, 2))
abc_dist = []
print('Simulating ABC paths from sampled parameters...')
for i, row in enumerate(tqdm(np.array(df))):
    m1, m2, m3, v0 = row[0], row[1], row[2], [row[3], row[4]]
    abc_dist.append([row[0], row[1], row[2], row[3], row[4]])
    sim = GravitySimulator(m1, m2, m3, v0) # no random noise on samples
    traj = sim.run()
    abc_trajs[i] = traj

abc_mean_path = np.sum(abc_trajs * abc_weights[:,np.newaxis,np.newaxis], axis=0)

extrema = get_extrema(abc_mean_path, extrema)

fig, axes = plt.subplots(figsize=(5,3), dpi=300)
# mean_sbi_params = np.mean(sbi_data, axis=0)
print('Simulating SBI paths from sampled parameters...')
for i, sample in enumerate(tqdm(sbi_data)):
    m1, m2, m3, v0 = sample[0], sample[1], sample[2], [sample[3], sample[4]]
    sim = GravitySimulator(m1, m2, m3, v0) # no random noise on samples
    traj = sim.run()
    sbi_paths[i] = traj

mean_sbi_path = np.mean(sbi_paths, axis=0)

# sim = GravitySimulator(mean_sbi_params[0], mean_sbi_params[1], mean_sbi_params[2], [mean_sbi_params[3], mean_sbi_params[4]])
# mean_sbi_path = sim.run()

extrema = get_extrema(mean_sbi_path, extrema)

alpha_val = 0.7
#axes.set_facecolor((0.9,0.9,0.9))

adjustment_factor = TRAJECTORY_MAGNITUDE_ADJUSTMENT_FACTOR
axes.scatter(observed_points[:,0]*adjustment_factor, observed_points[:,1]*adjustment_factor, color='black', zorder=10, marker='*', label='Observed Points')
sim.set_traj(prior_means_path)
sim.plot_traj(fig=fig,
              axes=axes,
              make_colorbar=False,
              save=False,
              cmap=plt.get_cmap('Greys').reversed(),
              color='grey',
              fade_lines=False,
              alpha=alpha_val,
              linestyle='-.',
              linewidth=1,
              label='Unbiased Prior')
sim.set_traj(mean_sbi_path)
sim.plot_traj(fig=fig,
              axes=axes,
              make_colorbar=False,
              save=False,
              cmap=plt.get_cmap('Greens').reversed(),
              color=colors[0],
              fade_lines=False,
              linewidth=1,
              alpha=alpha_val,
              label='SPLE')
sim.set_traj(abc_mean_path)
sim.plot_traj(fig=fig,
              axes=axes,
              make_colorbar=False,
              save=False,
              cmap=plt.get_cmap('Purples').reversed(),
              color=colors[1],
              fade_lines=False,
              linewidth=1,
              alpha=alpha_val,
              label='ABC')
sim.set_traj(true_path)
sim.plot_traj(fig=fig,
              axes=axes,
              make_colorbar=False,
              save=False,
              cmap=plt.get_cmap('Reds').reversed(),
              color='black',
              fade_lines=False,
              alpha=alpha_val,
              linestyle=':',
              linewidth=1,
              label='True Path',
              label_attractors=False)
sim.set_traj(maxent_weighted_average_path)
sim.plot_traj(fig=fig,
              axes=axes,
              make_colorbar=False,
              save=False,
              cmap=plt.get_cmap('Oranges').reversed(),
              color=colors[2],
              fade_lines=False,
              alpha=alpha_val,
              linestyle='-',
              linewidth=1,
              label='MaxEnt',
              label_attractors=True)

axes.set_xlim(-5, 130)# plt.xlim(extrema[0], extrema[1])
axes.set_ylim(-30, 75)# plt.ylim(extrema[2], extrema[3])

column_names = ['m1', 'm2', 'm3', 'v0x', 'v0y']
# plot the posterior distributions
abc_dist = np.array(abc_dist)
abc_frame = pd.DataFrame(abc_dist, columns = column_names)
sbi_dist = np.array(sbi_data)
sbi_frame = pd.DataFrame(sbi_dist, columns = column_names)
maxent_dist = np.load('maxent_prior_samples.npy')
maxent_frame = pd.DataFrame(maxent_dist, columns = column_names)

plt.legend(loc='upper left', bbox_to_anchor=(1.05,1.))
plt.tight_layout()
plt.savefig('paths_compare.png')
plt.savefig('paths_compare.svg')

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(5,5), dpi=300, sharex=True)

# iterate over the five parameters
legend = False
n_bins = 30
for i, key in enumerate(column_names):
    if i == len(column_names) - 1:
        legend = True
    sns.histplot(data=sbi_frame, x=key, ax=axes[i], color=colors[0],  stat='probability', element='step', kde=True, fill=False, bins=n_bins, lw=0.0)
    sns.histplot(data=abc_frame, x=key, ax=axes[i], color=colors[1], stat='probability', element='step', kde=True, fill=False, bins=n_bins, weights=abc_weights, lw=0.)
    sns.histplot(data=maxent_frame, x=key, ax=axes[i], color=colors[2],  stat='probability', element='step', kde=True, fill=False, bins=n_bins, weights=maxent_weights, lw=0.0)
    axes[i].axvline(prior_means[i], ls='-.', color='grey', lw=1.2)
    axes[i].axvline(true_params[i], ls='--', color='black', lw=1.2)
    axes[i].set_xlabel(key)
custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                Line2D([0], [0], color=colors[1], lw=4),
                Line2D([0], [0], color=colors[2], lw=4),
                Line2D([0], [0], color='black', ls='--', lw=4),
                Line2D([0], [0], color='grey', ls='-.', lw=4)]
axes[0].legend(custom_lines, ['SPLE', 'ABC', 'MaxEnt', 'True Parameter', 'Prior Mean'], loc='upper left', bbox_to_anchor=(1.05,1.))
plt.tight_layout()

plt.savefig('posterior_compare.png')
plt.savefig('posterior_compare.svg')


# time for cross-entropies

def get_crossent(prior_samples, posterior_samples, epsilon = 1e-7, x_range=[-100, 100], nbins=40, post_weights=None):
    prior_dists = []
    posterior_dists = []
    crossents = []
    for i in range(5):
        prior_dist, _ = np.histogram(prior_samples[:,i], bins=nbins, range=x_range, density=True)
        prior_dists.append(prior_dist)
        posterior_dist, _ = np.histogram(posterior_samples[:,i], bins=nbins, range=x_range, density=True, weights=post_weights)
        posterior_dists.append(posterior_dist)
        crossents.append( np.log(posterior_dist+epsilon) * (prior_dist + epsilon) )
    return -np.sum(crossents)


abc_prior = np.random.multivariate_normal(mean=prior_means, cov=np.eye(5)*2.5, size=abc_dist.shape[0])
abc_crossent = get_crossent(abc_prior, abc_dist, post_weights=abc_weights)

sbi_prior = np.random.multivariate_normal(mean=prior_means, cov=np.eye(5)*2.5, size=sbi_dist.shape[0])
sbi_crossent = get_crossent(sbi_prior, sbi_dist)

maxent_prior = np.random.multivariate_normal(prior_means, np.eye(5)*50, size=2048)
maxent_crossent = get_crossent(maxent_prior, maxent_prior, post_weights=maxent_weights)
print(f'CROSS-ENTROPY:\nABC: {abc_crossent}\nSBI: {sbi_crossent}\nMaxEnt: {maxent_crossent}')
crossent_values = [abc_crossent, sbi_crossent, maxent_crossent]
np.savetxt('crossent_values.txt', np.array(crossent_values), header='ABC, SBI, MaxEnt')