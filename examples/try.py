from MaxEntEp.functions import *
from MaxEntEp.SIR_model import *
import numpy as np

# SIR model
S_zero, I_zero, R_zero = 0.999, 0.001, 0.0  # same starting amount exposed
SIR_params = {'alpha': 0.003,  # recovery rate
              'beta': 0.02}  # infection rate (per contact?)

N = 2000

model = SIR_model([S_zero, I_zero, R_zero], SIR_params, N)
path = model()
model.plot_trajectory()

path[500]


e = 'laplace'
e_params = {'sigma': 0.0005}
eta = 1.5
def a(): return np.abs(np.random.normal(loc=0.0025, scale=0.005))
def b(): return np.abs(np.random.normal(loc=0.017, scale=0.004))
def g(x): return x[500][1]


reweighter = SIR_maxent_reweighter(
    200, 2000, g, [0.5617289], a, b, e, e_params, 10000, eta, 0.01)

results, grad_terms, g_k_vals = reweighter.run(debug=True)

traj = reweighter.make_plot(results, fixed_points=[[500, 0.5617289]])
plt.show()
