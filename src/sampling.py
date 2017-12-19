from pyDOE import *
import  numpy as np



def lhs_init(dim, samples, bounds):
    """
    https://pythonhosted.org/pyDOE/randomized.html#more-information
    :param dim: 
    :param samples: 
    :param bounds: 
    :return: 
    """
    x  = lhs(dim, samples=samples)
    x1range = bounds[0]
    x2range = bounds[1]
    x1 = x1range[0] + x[:,0]*(x1range[1]-x1range[0])
    x2 = x2range[0] + x[:,1]*(x2range[1]-x2range[0])
    x1= x1.reshape(samples, 1)
    x2= x2.reshape(samples, 1)
    x0 = np.concatenate((x1, x2), axis=1)
    return x0

def sample_training_data(fun, dim, samples, bounds):
    """
    
    :param fun: 
    :param dim: 
    :param samples: 
    :param bounds: 
    :return: 
    """
    x0 = lhs_init(dim, samples, bounds)
    y0 = []
    for x in x0:
        y0.append(fun(x))
    y0 = np.array(y0).reshape(samples,1)
    return x0, y0
