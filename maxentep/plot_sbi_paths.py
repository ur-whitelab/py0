import numpy as np
import matplotlib.pyplot as plt

from sbi_gravitation import GravitySimulator

data = np.genfromtxt('samples.txt')

for i, sample in enumerate(data):
    m1, m2, v0, x0 = sample[0], sample[1], [sample[2], sample[3]], [sample[4], sample[5]]
    sim = GravitySimulator(m1, m2, v0, x0) # no random noise on samples
    traj = sim.run()
    fig, axes = plt.subplots()
    if i != len(data) - 1:
        sim.plot_traj(fig=fig, axes=axes, save=False)
    else:
        sim.plot_traj(fig=fig, axes=axes, save=True, name='composite_traj.png', alpha=0.2)

print(data)