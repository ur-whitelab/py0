
import unittest
import numpy as np
import scipy.stats as ss


class TestFunctionality(unittest.TestCase):

    def test_imports(self):
        try:
            import maxentep 
            from maxentep import SIR_model, traj_quantile, reweight_laplace
        except ImportError as error:
            raise error
    
class TestMethod(unittest.TestCase):
    def test_SIR_model(self):
        # Testing for zero epidemiology parameters
        from maxentep import SIR_model
        N = 100
        L = 30
        trajs = np.empty((N, L, 3))
        alphas = np.zeros(N)
        betas = np.zeros(N)
        infected = ss.beta.rvs(1, 10, size = N)
        susceptible = ss.beta.rvs(5, 2, size = N)
        for i, a, b, I, S in zip(range(N), alphas, betas, infected, susceptible):
            model = SIR_model([S, I, 1 - S - I], a, b, L)
            trajs[i] = model()
        grad = np.gradient(trajs[:, :, 1], axis=1)
        assert np.count_nonzero(grad, axis = None) == 1

if __name__ == '__main__':
    unittest.main()
