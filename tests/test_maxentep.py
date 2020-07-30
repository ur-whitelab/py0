
import unittest
import numpy as np
import scipy.stats as ss


class TestFunctionality(unittest.TestCase):

    def test_imports(self):
        try:
            import maxentep
            from maxentep import SIRModel, traj_quantile, reweight_laplace
        except ImportError as error:
            raise error

class TestSIR(unittest.TestCase):
    def test_SIR_model(self):
        # Testing for zero epidemiology parameters
        from maxentep import SIRModel
        N = 100
        L = 30
        trajs = np.empty((N, L, 3))
        alphas = np.zeros(N)
        betas = np.zeros(N)
        infected = ss.beta.rvs(1, 10, size = N)
        susceptible = ss.beta.rvs(5, 2, size = N)
        for i, a, b, I, S in zip(range(N), alphas, betas, infected, susceptible):
            model = SIRModel([S, I, 1 - S - I], a, b, L)
            trajs[i] = model()
        grad = np.gradient(trajs[:, :, 1], axis=1)
        assert np.count_nonzero(grad, axis = None) == 1

class TestMetpop(unittest.TestCase):
    def test_metapop_model(self):
        from maxentep import MetaModel
        R = np.identity((3))
        T = np.identity(2) * 0.99
        T[0,1] = -0.01
        T[1,0] = -0.01
        start = np.random.random(size=(3, 2)) * 0.1
        model = MetaModel(start, R, T, lambda *_: [0.1, 0.1, 0.1])
        t1 = model.run(10)
if __name__ == '__main__':
    unittest.main()
