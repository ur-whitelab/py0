import pyabc
import numpy as np
import os
import tempfile
from sbi_gravitation import GravitySimulator, get_observation_points, prior_means

prior_cov = 50
prior = pyabc.Distribution(m1=pyabc.RV('norm', prior_means[0], prior_cov),
                            m2=pyabc.RV('norm', prior_means[1],prior_cov),
                            m3=pyabc.RV('norm', prior_means[2],prior_cov),
                            v0x=pyabc.RV('norm', prior_means[3],prior_cov),
                            v0y=pyabc.RV('norm', prior_means[4], prior_cov))

def distance(x, y):
    return(np.sum(np.linalg.norm(x['data'] - y['data'], axis=0)))

def model(parameter):
        '''params_dict should be: "m1", "m2", "m3", "v0"'''
        m1, m2, m3 = float(parameter['m1']), float(parameter['m2']), float(parameter['m3'])
        v0 = np.array([parameter['v0x'], parameter['v0y']])
        this_sim = GravitySimulator(m1, m2, m3, v0)
        this_traj = this_sim.run()
        summary_stats = get_observation_points(this_traj)
        return {'data': summary_stats}

db_path = 'sqlite:///' + os.path.join(os.getcwd(), 'abc_gravitation_wide_prior.db')        

if __name__ == '__main__':

    abc = pyabc.ABCSMC(models=model, parameter_priors=prior, distance_function=distance, population_size=100)

    observed_traj = np.genfromtxt('noisy_trajectory.txt')

    observation = get_observation_points(observed_traj)

    abc.new(db_path, {'data': observation})

    history = abc.run(minimum_epsilon=5., max_nr_populations=100)