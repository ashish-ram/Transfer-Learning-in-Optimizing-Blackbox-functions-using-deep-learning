from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from src.uncertainty import *


def get_expected_imp(x, *args):
    #print(x)
    X,Y, model, method= args
    y_hat = model.predict(x.reshape((1,2)), verbose=0)
    s_square = get_uncerainty(X,x, method)
    s=np.sqrt(s_square)
    E = EI(s,Y, y_hat)
    # y_min = np.min(Y)
    # if s==0:
    #     E=0
    # else:
    #     term = (y_min - y_hat)/s
    #     E = (y_min-y_hat)*norm.cdf(term)+s*norm.pdf(term)
    # #print("Expected improvement at {}:  {}".format(x, E))
    return -1*E


def EI(sigma, y, y_hat):
    s = sigma #np.sqrt(sigma)
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

def get_expected_imp_RBF(x, *args):
    X, y = args
    # Instanciate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    gp.fit(X, y)

    y_pred, sigma = gp.predict(x, return_std=True)
    sigma = sigma.reshape(y_pred.shape)

    E = EI(sigma, y, y_pred)

    return E


