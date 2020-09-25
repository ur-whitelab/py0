'''This script does a simple simulation of two gravitational attractors and returns their trajectory,
   then does SBI using that model.'''

import numpy as np
import matplotlib.pyplot as plt
from sbi.inference import infer
import sbi.utils as utils
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# colorline code from matplotlib examples https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

class GravitySimulator:
    def __init__(self, m1, m2, v0, x0, G=1.90809e5, dt=1e-3, nsteps=20000, random_noise=False):
        # only use one position/velocity, treat other body as origin --> do we need to correct for this?
        self.m1, self.m2, self.v0, self.x0, self.G, self.dt, self.nsteps = np.array(m1), np.array(m2), np.array(v0), np.array(x0), G, dt, nsteps
        self.positions = np.zeros([self.nsteps, 2])
        self.positions[0] = self.x0
        # first step special case
        self.positions[1] = self.x0 + self.v0 * self.dt + 0.5 * self.dt**2 * self.A(self.x0)
        self.iter_idx = 2
        self.random_noise = random_noise

    def rsquare(self, x):
        # square of distance between a point and origin
        return(x[0]**2 + x[1]**2)

    def A(self, x):
        # acceleration = Force/mass
        # F = G * m1 * m2 / r^2 + R(t) --> add random noise to force
        force = self.G * self.m1 * self.m2 / self.rsquare(x)
        # random force proportional to current magnitude
        # force += np.random.normal(loc=0., scale=np.abs(force)/500.) 
        # force is attractive, so multiply by unit vector toward origin
        force *= -1. * x / np.sqrt(np.sum(x**2))
        # compensate for one point always being at [0,0]
        return force / self.m2 - force/self.m1 
    
    def run(self):
        while(self.iter_idx < self.nsteps):
            self.step()
        if self.random_noise:
            self.positions = np.random.normal(self.positions, 0.8)
        return self.positions

    def step(self):
        # single step of integration with velocity verlet
        last_last_x = self.positions[self.iter_idx-2] 
        last_x = self.positions[self.iter_idx-1]
        self.positions[self.iter_idx] = 2 * last_x - last_last_x + self.A(last_x) * self.dt**2
        self.iter_idx += 1

    def plot_traj(self, name='trajectory.png', fig=None, axes=None, save=True, alpha=0.5):
        if fig is None and axes is None:
            fig, axes = plt.subplots()
        x, y =self.positions[:,0], self.positions[:,1]
        lc = colorline(x, y, alpha=alpha, cmap=plt.get_cmap('copper'))
        fig.colorbar(lc)
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        #plt.scatter(self.positions[:,0], self.positions[:,1], s=0.2)
        if save:
            plt.savefig(name)

    def set_traj(self, trajectory):
        self.positions = trajectory

if __name__ == 'main':

    m1 = 100. #solar masses
    m2 = 50. #solar masses
    G = 1.90809e5 #solar radius / solar mass * (km/s)^2
    # F = G * m1 * m2 / r^2

    v0 = np.array([100.,-70.]) # km/s (?)
    x0 = np.array([50.,30.5]) #solar radii

    sim = GravitySimulator(m1, m2, v0, x0, random_noise=True)
    traj = sim.run()
    print(traj)
    np.savetxt('trajectory.txt', traj)
    sim.plot_traj()

    observation_summary_stats = traj[:3001:500].flatten() # every 100th point in time only

    def sim_wrapper(params_list):
        '''params_list should be: m1, m2, v0[0], v0[1], x0[0], x0[1] in that order'''
        m1, m2 = float(params_list[0]), float(params_list[1])
        #v0 = v0#np.array([params_list[2], params_list[3]], dtype=np.float64)
        #x0 = x0#np.array([params_list[4], params_list[5]], dtype=np.float64)
        this_sim = GravitySimulator(m1, m2, v0, x0)
        this_traj = this_sim.run()
        summary_stats = torch.tensor(this_traj[:3001:500].flatten())
        return summary_stats

    prior_mins =  [10., 10., -100., -100., 10., 10.]#[10., 10.]#-150. * np.ones(40)
    prior_maxes =  [200., 200., 200., 200., 100., 100.]#150. * np.ones(40)#[200., 200.]

    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_mins, dtype=torch.float64),
                                        high=torch.as_tensor(prior_maxes, dtype=torch.float64))

    posterior = infer(sim_wrapper, prior, method='SNLE', num_simulations=400, num_workers=16)

    print('inference done, starting sampling...')

    samples = posterior.sample((2000,),
                                x=observation_summary_stats)

    np.savetxt('samples.txt', np.array(samples))

    print('samlping done, plotting.')

    fig, axes = utils.pairplot(samples)
    plt.savefig('pairplot.png')