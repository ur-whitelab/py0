import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import maxentep
import matplotlib.pyplot as plt
from sbi_gravitation import GravitySimulator, sim_wrapper

true_path = np.genfromtxt('true_trajectory.txt')
noisy_path = np.genfromtxt('noisy_trajectory.txt')
observed_points = true_path[:101:20]#noisy_path[:101:20]
print(observed_points, observed_points.shape)

# restraint structure: [value, uncertainty, indices... ]
restraints = []
for i, point in enumerate(observed_points):
    value1 = point[0]
    value2 = point[1]
    uncertainty = 0.1
    index = 20 * i
    restraints.append([value1, uncertainty, index, 0])
    restraints.append([value2, uncertainty, index, 1])
print(f'restraint length: {len(restraints)}')

# prepare laplace restraints like in https://github.com/ur-whitelab/maxent-epidemiology/blob/master/examples/SIR_Example.ipynb
laplace_restraints = []

for i in range(len(restraints)):
    traj_index = tuple(restraints[i][2:])
    value = restraints[i][0]
    uncertainty = restraints[i][1]
    p = maxentep.Laplace(uncertainty)
    #p = maxentep.EmptyPrior()
    r = maxentep.Restraint(lambda traj, i=traj_index: traj[i], value, p)
    laplace_restraints.append(r)

true_params = [100., 50., 75., 15.,-40.]

prior_means = [85., 40., 70., 12., -30.]#[99.9, 49.9, 75.1, 14.9, -39.9]

prior_cov = np.eye(5) * 2.5

prior_dist = np.random.multivariate_normal(prior_means, prior_cov, size=3000)

print(prior_dist, prior_dist.shape)
print(f'the average of 10k samples from numpy is {np.mean(prior_dist, axis=0)}')

trajs = np.zeros([prior_dist.shape[0], 100, 2])

if os.path.exists('maxent_raw_trajectories.npy'):
    trajs = np.load('maxent_raw_trajectories.npy')
else:
    print('sampling trajectories...')
    for i, sample in enumerate(tqdm(prior_dist)):
        m1, m2, m3, v0 = sample[0], sample[1], sample[2], sample[3:]
        sim = GravitySimulator(m1, m2, m3, v0)
        traj = sim.run()
        trajs[i] = traj

np.save('maxent_raw_trajectories.npy', trajs)    

model = maxentep.MaxentModel(laplace_restraints)
model.compile(tf.keras.optimizers.Adam(1e-5), 'mean_squared_error')
h = model.fit(trajs, batch_size=128, epochs=10000, verbose=1)
plt.plot(h.history['loss'])
plt.savefig('maxent_loss.png')

weights = model.traj_weights

print(weights)

np.savetxt('maxent_traj_weights.txt', weights)
