import  numpy as np
from functions import functions as f
import airfoil.utility as u
import time
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from skopt import gp_minimize

start_time = time.time()
def objective(x):
    y = funObj.objective_6D(x)
    return y
for i in range(1):
    test_dir = r'./Airfoil6D_data/opt_SKOPT_LB_Mach2_{}/logs/'.format(i)
    u.make_sure_path_exists(test_dir)
    funObj = f.Airfoil(3, 0.77971043, 5e005, test_dir, False)
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    res = gp_minimize(objective,  # the function to minimize
                      bounds,  # the bounds on each dimension of x
                      acq_func="LCB",  # the acquisition function
                      n_calls=200,  # the number of evaluations of f
                      n_random_starts=25)  # ,  # the number of random initialization points
    # noise=0.1**2,       # the noise level (optional)
    # random_state=1)   # the random
    np.savez(r'./Airfoil6D_data/opt_SKOPT_LB_mach2_{}/Airfoil6D_XY_SKOPT_LB.npz'.format(i), np.array(res.x_iters), res.func_vals,
             time.time() - start_time)
    # xy  = np.load(r'./Airfoil6D_data/opt_SKOPT/Airfoil6D_XY_SKOPT.npz')
    print('run time: ', time.time() - start_time)

for i in range(1):
    test_dir = r'./Airfoil6D_data/opt_SKOPT_PI_Mach2_{}/logs/'.format(i)
    u.make_sure_path_exists(test_dir)
    funObj = f.Airfoil(3, 0.77971043, 5e005, test_dir, False)
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    res = gp_minimize(objective,  # the function to minimize
                      bounds,  # the bounds on each dimension of x
                      acq_func="PI",  # the acquisition function
                      n_calls=200,  # the number of evaluations of f
                      n_random_starts=25)  # ,  # the number of random initialization points
    # noise=0.1**2,       # the noise level (optional)
    # random_state=1)   # the random
    np.savez(r'./Airfoil6D_data/opt_SKOPT_PI_Mach2_{}/Airfoil6D_XY_SKOPT_PI.npz'.format(i), np.array(res.x_iters), res.func_vals,
             time.time() - start_time)
    # xy  = np.load(r'./Airfoil6D_data/opt_SKOPT/Airfoil6D_XY_SKOPT.npz')
    print('run time: ', time.time() - start_time)