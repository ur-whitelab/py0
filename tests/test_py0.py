
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
