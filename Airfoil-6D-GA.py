import  numpy as np
from functions import functions as f
import airfoil.utility as u
import time
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from skopt import gp_minimize

X = np.array([[0,0,0,0,0,0]])
Y = np.array([0])
def objective(x):
    global X, Y
    y = funObj.objective_6D(x)
    X = np.concatenate((X, np.array(x).reshape(1,6)))
    Y = np.concatenate((Y, np.array([y])))
    return y

test_dir = r'./Airfoil6D_data/opt_GA/logs/'
u.make_sure_path_exists(test_dir)
funObj = f.Airfoil(3, 0.2, 5e005, test_dir, False)
bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]

start_time = time.time()

ret = differential_evolution(objective, bounds)

np.savez(r'./Airfoil6D_data/opt_GA/Airfoil6D_XY_GA.npz',X, Y, time.time() - start_time)
#xy  = np.load(r'./Airfoil6D_data/opt_SKOPT/Airfoil6D_XY_SKOPT.npz')
print('run time: ', time.time() - start_time)
