import numpy as np
import matplotlib.pyplot as plt

trajs = np.load('maxent_raw_trajectories.npy')

fig, axes = plt.subplots()
for traj in trajs:
    axes.plot(traj[:,0], traj[:,1], alpha=0.1, color='blue')
plt.xlim(-5, 130)# plt.xlim(extrema[0], extrema[1])
plt.ylim(-30, 75)# plt.ylim(extrema[2], extrema[3])

plt.savefig('maxent_raw_trajectories.png')