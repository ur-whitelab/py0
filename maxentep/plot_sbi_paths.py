from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pyabc
from pyabc import ABCSMC
from sbi_gravitation import GravitySimulator
from abc_gravitation import db_path, distance, model, prior

import seaborn as sns
sns.set_context('paper')
sns.set_style('darkgrid',  {'xtick.bottom':True, 'ytick.left':True, 'xtick.color': '#333333', 'ytick.color': '#333333'})
#plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']


all_paths = []

true_path = np.genfromtxt('true_trajectory.txt')
noisy_path = np.genfromtxt('noisy_trajectory.txt')
sbi_data = np.genfromtxt('samples.txt')

maxent_paths = np.load('maxent_raw_trajectories.npy')
maxent_weights = np.genfromtxt('maxent_traj_weights.txt')

plt.figure()
plt.plot(maxent_weights)
plt.savefig('maxent_weights.png')
plt.close()

maxent_weighted_average_path = np.average(maxent_paths, axis=0, weights=maxent_weights)
print(f'maxent weighted average path shape: {maxent_weighted_average_path.shape}')

x_min, y_min = 1000000., 100000.
x_max, y_max = -1000000., -100000.

all_paths.append(true_path)
all_paths.append(noisy_path)

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

observed_points = noisy_path[:101:20]

prior_means = [85., 40., 70., 12., -30.]#[99.9, 49.9, 75.1, 14.9, -39.9]

sim = GravitySimulator(prior_means[0], prior_means[1], prior_means[2], prior_means[3:])
prior_means_path = sim.run()

extrema = get_extrema(prior_means_path, extrema)

abc_continued = ABCSMC(model, prior, distance)
abc_continued.load(db_path)
history = abc_continued.history
df, w = history.get_distribution(m=0, t=history.max_t)
param_means = []
for column in df.columns:
    param_means.append(np.mean(df[column]))
m1, m2, m3, v0 = param_means[0], param_means[1], param_means[2], [param_means[3], param_means[4]]
sim = GravitySimulator(m1, m2, m3, v0) # no random noise on samples
abc_mean_path = sim.run()

extrema = get_extrema(abc_mean_path, extrema)


fig, axes = plt.subplots(figsize=(5,3))
mean_sbi_params = np.mean(sbi_data, axis=0)
# for i, sample in enumerate(tqdm(sbi_data)):
#     m1, m2, m3, v0 = sample[0], sample[1], sample[2], [sample[3], sample[4]]
#     sim = GravitySimulator(m1, m2, m3, v0) # no random noise on samples
#     traj = sim.run()
#     sbi_paths[i] = traj
    # if i != len(sbi_data) - 1:
    #     sim.plot_traj(fig=fig, axes=axes, save=False, alpha=0.15)
    # else:
    #     plt.scatter(sim.attractor_positions[:,0], sim.attractor_positions[:,1], color='black', zorder=1000)
    #     sim.plot_traj(fig=fig, axes=axes, make_colorbar=True, save=True, name='composite_traj.png', alpha=0.2)

# mean_sbi_path = np.mean(sbi_paths, axis=0)
# print(mean_sbi_path.shape)
m1, m2, m3, v0 = mean_sbi_params[0], mean_sbi_params[1], mean_sbi_params[2], [mean_sbi_params[3], mean_sbi_params[4]]
sim = GravitySimulator(m1, m2, m3, v0)
mean_sbi_path = sim.run()

extrema = get_extrema(mean_sbi_path, extrema)

alpha_val = 0.7
#axes.set_facecolor((0.9,0.9,0.9))

axes.scatter(observed_points[:,0], observed_points[:,1], color='black', zorder=10, marker='*', label='Observed Points')
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
              label='Unbiased Prior')
sim.set_traj(mean_sbi_path)
sim.plot_traj(fig=fig,
              axes=axes,
              make_colorbar=False,
              save=False,
              cmap=plt.get_cmap('Greens').reversed(),
              color=colors[0],
              fade_lines=False,
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
              label='MaxEnt',
              label_attractors=True)

plt.xlim(-5, 130)# plt.xlim(extrema[0], extrema[1])
plt.ylim(-30, 75)# plt.ylim(extrema[2], extrema[3])
plt.legend(loc='upper left', bbox_to_anchor=(1.05,1.))
plt.tight_layout()
plt.savefig('paths_compare.png')