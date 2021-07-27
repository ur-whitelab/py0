
import unittest
import numpy as np
import scipy.stats as ss


class TestFunctionality(unittest.TestCase):

    def test_imports(self):
        try:
            import py0
            from py0 import SIRModel, traj_quantile
        except ImportError as error:
            raise error

class TestSIR(unittest.TestCase):
    def test_SIR_model(self):
        # Testing for zero epidemiology parameters
        import py0
        from py0 import SIRModel
        import sys
        N = 100
        L = 30
        trajs = np.empty((N, L, 3))
        alphas = np.zeros(N)
        betas = np.ones(N)
        infected = ss.beta.rvs(1, 10, size = N)
        susceptible = ss.beta.rvs(5, 2, size = N)
        for i, a, b, I, S in zip(range(N), alphas, betas, infected, susceptible):
            model = SIRModel([S, I, 1 - S - I], a, b, L)
            trajs[i] = model()
        grad = np.gradient(trajs[:, :, 1], axis=1)
        np.set_printoptions(threshold=sys.maxsize)
        print(trajs)
        print(np.count_nonzero(grad, axis=None) == 1)
        assert np.count_nonzero(grad, axis = None) == 1

class TestMetpop(unittest.TestCase):
    def test_metapop_model(self):
        import py0
        from py0 import MetaModel
        R = np.identity((3))
        T = np.identity(2) * 0.99
        T[0,1] = -0.01
        T[1,0] = -0.01
        start = np.random.random(size=(3, 2)) * 0.1
        infect_fxn = py0.contact_infection_func(
            [1])
        model = py0.MetaModel(infect_fxn, 10)
        beta_I = 0.05
        t1 = model(R, T,
                   start, np.array([beta_I]))[0]
if __name__ == '__main__':
    unittest.main()
