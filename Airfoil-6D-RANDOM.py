import  numpy as np
from functions import functions as f
import airfoil.utility as u
import time
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from skopt import dummy_minimize

start_time = time.time()
def objective(x):
    y = funObj.objective_6D(x)
    return y

test_dir = r'./Airfoil6D_data/opt_RANDOM/logs/'
u.make_sure_path_exists(test_dir)
funObj = f.Airfoil(3, 0.2, 5e005, test_dir, False)
bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
res = dummy_minimize(objective,                  # the function to minimize
                  bounds,      # the bounds on each dimension of x
                  n_calls=225)#,         # the number of evaluations of f
                 # random_state=1)   # the random
np.savez(r'./Airfoil6D_data/opt_RANDOM/Airfoil6D_XY_RANDOM.npz',np.array(res.x_iters), res.func_vals, time.time() - start_time)
#xy  = np.load(r'./Airfoil6D_data/opt_RANDOM/Airfoil6D_XY_RANDOM.npz')
print('run time: ', time.time() - start_time)
