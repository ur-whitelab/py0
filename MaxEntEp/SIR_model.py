import numpy as np
import matplotlib.pyplot as plt
from MaxEntEp.functions import *

class SIR_model:
    def __init__(self, initial_compartments, params_dict, N_steps=1000):
        self.I, self.R = initial_compartments[1], initial_compartments[2]
        self.alpha, self.beta = params_dict['alpha'], params_dict['beta']
        self.trajectory = []
        self.N_steps = N_steps

    def step(self):
        I_next = min(1., max(0., self.I + self.beta * self.I *
                             (1.-self.I-self.R) - self.alpha * self.I))
        R_next = min(1., max(0., self.R + self.alpha * self.I))
        self.I, self.R = I_next, R_next

    def run(self):
        self.trajectory.append([1. - self.I - self.R, self.I, self.R])
        for i in range(self.N_steps-1):
            self.step()
            self.trajectory.append([1. - self.I - self.R, self.I, self.R])

    def get_trajectory(self):
        return np.array(self.trajectory)

    def plot_trajectory(self):
        traj = self.get_trajectory()
        plt.figure()
        x = range(self.N_steps)
        plt.plot(x, traj[:, 0], label='Susceptible')
        plt.plot(x, traj[:, 1], label='Infected')
        plt.plot(x, traj[:, 2], label='Recovered')
        plt.legend()

    def __call__(self):
        self.run()
        return(self.get_trajectory())


class SIR_maxent_reweighter:
    '''Uses  the SIR_model class to do maximum entropy with importance sampling.(?)
        Parameters:
            alpha_distro, beta_distro: callable. distribution for drawing alpha and beta values, respectively.
                should return single scalars.
            error_dist_type: string. name of prior distribution of the error in measurement of g_bar.
                for now, can only be 'laplace'. TODO: add more distros
            error_dist_params: dictionary. contains key-value pairs of parameters corresponding to error distro.
                for now, can only be for laplace, which needs 'sigma' (always centered at 0).
            N_param_samples: how many times to sample parameters. integer.
            N_timesteps: how many timesteps to run each model
            target_functions: callable or list of callables. The way to evaluate the target observable.
                Should take in the trajectory output by the SIR model and give back a scalar.
            target_function_values: scalar or list of scalars that target_functions should match.
            '''

    def __init__(self,
                 N_param_samples,
                 N_timesteps,
                 target_functions,
                 target_function_values,
                 alpha_distro,
                 beta_distro,
                 error_dist_type,
                 error_dist_params,
                 N_SGD_iterations,
                 learning_rate,
                 SGD_tolerance):
        if type(target_functions) is list and all([callable(a) for a in [type, callable]]):
            self.g = target_functions
        elif callable(target_functions):
            self.g = [target_functions]
        else:
            raise(
                TypeError('target_functions must be either a callable or list of callables'))
        self.target_vals = np.array(target_function_values)
        self.N_param_samples = N_param_samples
        self.N_timesteps = N_timesteps
        self.alpha_distro, self.beta_distro = alpha_distro, beta_distro
        self.error_dist_type = error_dist_type
        self.error_dist_params = error_dist_params
        self.N_iters = N_SGD_iterations
        self.tolerance = SGD_tolerance
        self.initial_proportions = [0.999, 0.001, 0.0]
        # per-coordinate learning rates, same initially
        self.learning_rate = learning_rate
        self.eta = self.learning_rate * np.ones(len(self.g))

    def run(self, debug=False):
        trajs, g_k_arrs = [], []
        sigma = self.error_dist_params['sigma']
        w_is = np.ones(self.N_param_samples)
        if debug:
          gradient_terms_arr = []
          g_k_vals_arr = []
        #pick N_param_samples parameters, get N trajectories
        for i in range(self.N_param_samples):
            traj = self.step()
            trajs.append(traj)
            #for each constraint, get g(f(theta)) for each trajectory
            g_k_arrs.append([g(traj) for g in self.g])
        # now get w_i functions
        lambda_ks = np.ones(len(self.g))  # start with 1.0 for each lambda_k
        iters = 0  # number of SGD iterations to do...
        print('Starting SGD...')
        expected_g_k_vals = np.zeros(len(self.g))
        grad_sums = np.zeros(len(lambda_ks))  # for adaptive learning rates
        while iters < self.N_iters and np.abs(expected_g_k_vals - self.target_vals) > self.tolerance:
            if iters % 100 == 0:
                print(f'SGD iteration {iters}')
            iters += 1
            for i in range(self.N_param_samples):
                w_is[i] = self.w_i(lambda_ks, g_k_arrs[i],
                                   self.error_dist_params['sigma'])
            #w_is = w_is/np.sum(w_is)
            for k in range(len(lambda_ks)):
                lambda_k_grad = 0.
                expected_g_k_val = 0.
                expected_squared_g_k_val = 0.
                # get E[g_k]
                for i in range(self.N_param_samples):
                    expected_g_k_val += w_is[i] * g_k_arrs[i][k]
                    expected_squared_g_k_val += w_is[i] * g_k_arrs[i][k]**2
                # normalize
                expected_g_k_val /= np.sum(w_is)
                expected_squared_g_k_val /= np.sum(w_is)
                # get Var[g_k]
                g_k_variance = expected_squared_g_k_val - expected_g_k_val**2
                term1 = (expected_g_k_val -
                         self.target_vals[k] + self.xi_k(lambda_ks[k], sigma))
                term2 = (self.d_xi_k_d_lambda_k(
                    lambda_ks[k], sigma) - g_k_variance)

                lambda_k_grad = term1 * term2
                grad_sums[k] += lambda_k_grad**2
                self.eta[k] = self.learning_rate / np.sqrt(grad_sums)
                if debug:
                  gradient_terms_arr.append(
                      [term1, term2, lambda_k_grad, lambda_ks[k]])
                  g_k_vals_arr.append([expected_g_k_val])
                lambda_ks[k] -= self.eta[k] * lambda_k_grad
                expected_g_k_vals[k] = expected_g_k_val
        # for each k (each constraint), do an optimization to find lambda_k
        #for g_ks in g_k_arrs:
        #    for k, g_k in enumerate(g_ks):
        result = {'lambda_ks': lambda_ks,
                  'weights': w_is, 'trajectories': trajs}
        if debug:
          return([result, gradient_terms_arr, g_k_vals_arr])
        return(result)

    def w_i(self, lambda_ks, g_ks, sigma):
        log_prod = 0.
        for k, g_k in enumerate(g_ks):
            log_prod += (-lambda_ks[k] * g_k) + \
                np.log(self.xi_k_denom(lambda_ks[k], sigma))
        return np.exp(log_prod)

    def xi_k(self, lambda_k, sigma):
        # for use in the optimization of lamda_k during reweighting
        # currently only for laplace prior on errors
        # sigma is the laplace sigma parameter
        return -1. * lambda_k * sigma**2 / (1. - lambda_k**2 * sigma**2 / 2)

    def d_xi_k_d_lambda_k(self, lambda_k, sigma):
        # derivative of xi_k w/ respect to lambda_k
        # wolfram alpha answer:
        # -2. * sigma**2 * (sigma**2 * lambda_k**2 + 2) / (2 - sigma**2 * lambda_k**2)**2
        return (1.5 - 1./(lambda_k**2 * sigma**2))

    def xi_k_denom(self, lambda_k, sigma):
        # for use in reweighting step
        # currently only for laplace prior on errors
        # sigma is the laplace sigma parameter
        return (1. / (lambda_k + np.sqrt(2)/sigma) + 1. / (np.sqrt(2)/sigma - lambda_k))

    def step(self):
        # sample parameters from alpha and beta distros
        # for now, always start with the same S, I, R
        # same starting amount exposed
        alpha, beta = self.alpha_distro(), self.beta_distro()
        SIR_params = {'alpha': alpha,  # recovery rate
                      'beta': beta}  # infection rate

        model = SIR_model(self.initial_proportions,
                          SIR_params, self.N_timesteps)
        model.run()
        return model.get_trajectory()

    def make_plot(self, results, fixed_points=[], alpha_val=0.2, plot_means=True):
      '''Make a plot of all the trajectories and the average trajectory based on
      parameter weights.'''

      w = results['weights']
      w /= np.sum(w)

      x = range(self.N_timesteps)
      trajs = np.array(results['trajectories'])
      # weighted quantiles doesn't support axis, unfortanutely
      # fake it using apply_along
      qtrajs = np.apply_along_axis(lambda x: weighted_quantile(
          x, [1/3, 1/2, 2/3], sample_weight=w), 0, trajs)
      if plot_means:
        # approximate quantiles as distance from median applied to mean
        mtrajs = np.sum(trajs * w[:, np.newaxis, np.newaxis], axis=0)
        qtrajs[0, :, :] = qtrajs[0, :, :] - qtrajs[1, :, :] + mtrajs
        qtrajs[2, :, :] = qtrajs[2, :, :] - qtrajs[1, :, :] + mtrajs
        qtrajs[1, :, :] = mtrajs

      plt.figure()
      plt.xlabel('Timestep')
      plt.ylabel('Fraction of Population')
      for i, n in enumerate(['Susceptible', 'Infected', 'Recovered']):
        plt.plot(x, qtrajs[1, :, i], color=f'C{i}', label=n)
        plt.fill_between(x, qtrajs[0, :, i], qtrajs[-1, :, i],
                         color=f'C{i}', alpha=alpha_val)
      plt.plot(x, np.sum(qtrajs[1, :, :], axis=1),
               color='gray', label='Total', linestyle=':')
      for point in fixed_points:
        plt.scatter(point[0], point[1], color='black', label='Target point(s)')
      # add margin for legend
      plt.xlim(0, max(x) * 1.2)
      plt.legend(loc='center right')
      return(mtrajs)
