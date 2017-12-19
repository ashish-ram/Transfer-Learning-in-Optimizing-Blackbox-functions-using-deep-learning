import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from numpy import linalg as LA

class Acquisition(object):
    def __init__(self, X, y,  model, kernel=None):
        self.X = X
        self.y = y
        self.model = model
        if kernel is None:
            # Instanciate a Gaussian Process model
            self.kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        self.gp.fit(X, y)

    def EI(self, sigma, y, y_hat):
        s = sigma  # np.sqrt(sigma)
        y_min = y.min()
        E = np.zeros_like(y_hat)
        mask = s > 0
        improved = y_min - y_hat[mask]
        scaled = improved / s[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improved * cdf
        explore = s[mask] * pdf
        E[mask] = exploit + explore

        return E

    def get_expected_imp_RBF(self, x, *args):
        _, sigma = self.gp.predict(x, return_std=True)
        y_hat = self.model.predict(x)
        sigma = sigma.reshape(y_hat.shape)
        E = self.EI(sigma, self.y, y_hat)

        return E, sigma

    def get_expected_imp_distance(self, x, *args):

        # TODO
        pass

    def get_uncertainty_distance(self,x):
        distance = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            distance[i] = np.min(LA.norm(self.X - x[i], axis=1))
        # print("distance: ",distance)
        # return distance
        # return np.sqrt(distance)
        return np.power(distance, 1)


    def get_lower_bound_distance(self, x, *args):
        # print("shape of x: ",x.shape)
        # print(type(x))
        x = np.array(x).reshape(1, x.shape[0])
        A  = args[0]
        #print(A)
        y_hat = self.model.predict(x, verbose=0)
        s = self.get_uncertainty_distance(x)
        LB = (1-A)*y_hat -A*s
        return LB
