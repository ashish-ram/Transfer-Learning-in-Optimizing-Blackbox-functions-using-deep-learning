import pylab
from src import helper as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from skopt import gp_minimize
from functions import functions as f
import airfoil.utility as u
import numpy.random as random



test_dir = r'./Airfoil_data/SKOPT/'
u.make_sure_path_exists(test_dir)
airfoil = f.Airfoil(3, 0.2, 5e005, test_dir, False)
def objective(x):
    y = airfoil.objective_opt(x)
    return y

itr_num = 2
for i in range(itr_num):
    initial_samples =1# 5 + random.randint(3)
    print('Starting iteration number : ', i)
    print('initial number of samples: ', initial_samples)
    res = gp_minimize(objective,  # the function to minimize
                      airfoil.bounds,  # the bounds on each dimension of x
                      acq_func="LCB",  # the acquisition function
                      n_calls=2,  # the number of evaluations of f
                      n_random_starts=initial_samples,  # the number of random initialization points
                      # noise=0.1**2,       # the noise level (optional)
                      random_state=random.randint(500)
                      )  # the random seed
    X = np.array(res.x_iters)
    Y = np.array(res.func_vals)
    np.savez(test_dir+r'Airfoil_XY_SKOPT_itr_{}'.format(i), X, Y)



