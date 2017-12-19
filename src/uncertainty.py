import numpy as np
from numpy import linalg as LA

def get_uncerainty(x, *params):
    X, method = params
    if method is 'distance':
        uncertainty = get_uncerainty_distance(X, x)
    return uncertainty


def get_uncerainty_distance(X, x):
    dim = len(x)
    uncertainty = np.min(LA.norm(X - x, axis=1))

    # print("uncertainty: ",uncertainty)
    return uncertainty
