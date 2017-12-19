import numpy.random as random
import errno
import pylab
from functions import functions as f
from src import sampling
from src import Acquisition
import keras
from keras.models import Sequential
from keras.layers import Dense
from pyDOE import *
import airfoil.utility as u
import time
from functions import functions
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from keras.callbacks import EarlyStopping

initial_samples = 25
infill_counter = 200
A = 0.92
epochs = 1000
patience = 1000

graphics = False
path = r'./AirfoilGen_test/'


def setupANN(dim, units, loss):
    model = Sequential()

    ## making the model graph, Stacking layers is done by .add():
    model.add(Dense(units=units, input_dim=dim, activation='sigmoid'))
    model.add(Dense(units=units, activation='sigmoid'))
    model.add(Dense(units=units, activation="sigmoid"))
    model.add(Dense(units=units, activation="sigmoid"))
    model.add(Dense(units=1, activation='linear'))

    # optmiser = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optmiser = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)

    # configure the model's learning process; loss and optimisation etc
    model.compile(loss=loss,
                  optimizer=optmiser, metrics=["mae"])

    return model



def single_run(funObj, initial_samples, model_old=None, X_old=None, Y_old=None):
    global infill_counter
    global A
    global patience
    global epochs
    global graphics


    if model_old==None and X_old==None and Y_old==None:
        ## Generate 6D sampling data:
        print('starting from fresh...')
        dim = 6
        X = lhs(n=dim, samples=initial_samples)
        X = X - 1
        Y = []
        for xx in X:
            ld = funObj.objective_6D(xx)  # use for 6D LD test
            Y.append(ld)
        Y = np.array(Y)
        a = np.array([funObj.params[0:3]] * X.shape[0])
        X = np.concatenate((X, a), axis=1)

    else:
        print('Locating the first infill for the new context using the previous context ...')
        EIobj = Acquisition.Acquisition(X=X_old, y=Y_old, model=model_old)
        bounds = [(-0.1, 0.2), (-0.1, 0.5), (-0.1, 0.5), (-0.5, 0.1), (-0.5, 0.1), (-0.2, 0.1),
                  (funObj.params[0], funObj.params[0]),
                  (funObj.params[1], funObj.params[1]),
                  (funObj.params[2], funObj.params[2])]


        ret = differential_evolution(EIobj.get_lower_bound_distance, bounds, args=(A, 0.1)) #0.1 is just a placeholder

        first_x = ret.x
        first_y = np.array([funObj.objective_6D(first_x[0:6])])
        first_x = first_x.reshape(1, 9)
        # next_y = next_y.reshape(1, 1)
        X = np.concatenate((X_old, first_x))
        Y = np.concatenate((Y_old, first_y))

    #conv = IsConverged()
    print("X: ", X)
    print("Y: ", Y)
    print('Optimising the new context')
    for i in range(infill_counter):  # number of infills
        print('==========> infill {}'.format(i))
        dim = 6 + 3
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='min')

        model = setupANN(dim=dim, units=200, loss='mean_squared_error')
        hist = model.fit(X, Y, epochs=5000, batch_size=100, verbose=0)#, validation_split=0.15, callbacks=[early_stopping])

        EIobj = Acquisition.Acquisition(X=X, y=Y, model=model)

        # x0 = [0.05, 0.1, 0.2, -0.2, -0.1, -0.05]
        # x0 = np.array(x0).reshape(1,6)
        # ret = basinhopping(EIobj.get_lower_bound_distance, x0, niter=20, minimizer_kwargs={"args": 0.92})

        bounds = [(-0.1, 0.2), (-0.1, 0.5), (-0.1, 0.5), (-0.5, 0.1), (-0.5, 0.1), (-0.2, 0.1),
                  (funObj.params[0], funObj.params[0]),
                  (funObj.params[1], funObj.params[1]),
                  (funObj.params[2], funObj.params[2])]

        ret = differential_evolution(EIobj.get_lower_bound_distance, bounds, args=(A,0.1))


        next_x = ret.x
        next_y = np.array([funObj.objective_6D(next_x[0:6])])

        next_x = next_x.reshape(1, 9)
        #next_y = next_y.reshape(1, 1)
        X = np.concatenate((X, next_x))
        Y = np.concatenate((Y, next_y))

        plt.plot(hist.history['loss'], label='loss')
        #plt.plot(hist.history['val_loss'], label='val_loss')
        #plt.plot(hist.history['val_mean_absolute_error'], label='val_mae')
        plt.plot(hist.history['mean_absolute_error'], label='mae')
        plt.legend()






        #plot(funObj, X, hist, zz_pred, LB, s)
        ##### lool over ########

    print('minima found is f(x) = {}'.format(Y.min()))
    return model, X, Y

# start = time.time()
# test_dir = r'./Airfoil6D_data/opt/logs/'
# u.make_sure_path_exists(test_dir)
# funObj = f.Airfoil(3, 0.2, 5e005, test_dir, True)
import os
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

make_sure_path_exists(path)

for i in range(1):  # how  many times end to end generalisation is done
    i = 0
    series = 0.2 + np.random.rand(5) * 0.2 #2=how many in a series of contexts
    np.savez(path + 'series_{}.npz'.format(i), series)

    start = time.time()

    logdir = path+r's{}/context{}/'.format(i,0)
    u.make_sure_path_exists(logdir)
    funObj = f.Airfoil(3, series[0], 5e005, logdir, graphics)


    print('Starting iteration number : ', i)
    print('initial number of samples: ', initial_samples)

    model, X, Y = single_run(funObj, initial_samples=initial_samples, model_old=None, X_old=None, Y_old=None)
    model.save(path + r"Airfoil_model_series_{}_context_{}".format(0,0))
    np.savez(path + r'/Airfoil_XY_series_{}_context_{}'.format(0,0),X,Y)

    for j in range(1,series.shape[0]):
        logdir = path + r's{}/context{}/'.format(i, j)
        u.make_sure_path_exists(logdir)
        funObj = f.Airfoil(3, series[j], 5e005, logdir, graphics)
        #initial_samples = 20 + random.randint(10)

        model, X, Y = single_run(funObj, initial_samples=initial_samples, model_old=model, X_old=X, Y_old=Y)

        model.save(path + r"Airfoil_model_series_{}_context_{}".format(i,j))
        np.savez(path + r'/Airfoil_XY_series_{}_context_{}'.format(i,j), X, Y)





#
# #######################  optimize using SKOPT  #############################
# import  numpy as np
# from functions import functions as f
# import airfoil.utility as u
# import time
# from scipy.optimize import basinhopping
# from scipy.optimize import differential_evolution
# from skopt import gp_minimize
#
# start_time = time.time()
# def objective(x):
#     y = funObj.objective_6D(x)
#     return y
#
# test_dir = r'./Airfoil6D_data/opt_SKOPT/logs/'
# u.make_sure_path_exists(test_dir)
# funObj = f.Airfoil(3, 0.2, 5e005, test_dir, graphics)
# bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
# res = gp_minimize(objective,                  # the function to minimize
#                   bounds,      # the bounds on each dimension of x
#                   acq_func="EI",      # the acquisition function
#                   n_calls=5,         # the number of evaluations of f
#                   n_random_starts=2,  # the number of random initialization points
#                   #noise=0.1**2,       # the noise level (optional)
#                   random_state=1)   # the random
# np.savez(r'./Airfoil6D_data/opt_SKOPT/Airfoil6D_XY_SKOPT.npz',np.array(res.x_iters), res.func_vals, time.time() - start_time)
# #xy  = np.load(r'./Airfoil6D_data/opt_SKOPT/Airfoil6D_XY_SKOPT.npz')
# print('run time: ', time.time() - start_time)
#
#
# #######################  optimize using random search  #############################
# import  numpy as np
# from functions import functions as f
# import airfoil.utility as u
# import time
# from scipy.optimize import basinhopping
# from scipy.optimize import differential_evolution
# from skopt import dummy_minimize
#
# start_time = time.time()
# def objective(x):
#     y = funObj.objective_6D(x)
#     return y
#
# test_dir = r'./Airfoil6D_data/opt_RANDOM/logs/'
# u.make_sure_path_exists(test_dir)
# funObj = f.Airfoil(3, 0.2, 5e005, test_dir, graphics)
# bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
# res = dummy_minimize(objective,                  # the function to minimize
#                   bounds,      # the bounds on each dimension of x
#                   n_calls=5,         # the number of evaluations of f
#                   random_state=1)   # the random
# np.savez(r'./Airfoil6D_data/opt_RANDOM/Airfoil6D_XY_RANDOM.npz',np.array(res.x_iters), res.func_vals, time.time() - start_time)
# #xy  = np.load(r'./Airfoil6D_data/opt_RANDOM/Airfoil6D_XY_RANDOM.npz')
# print('run time: ', time.time() - start_time)
#
#
# #######################  optimize using GA  #############################
# import  numpy as np
# from functions import functions as f
# import airfoil.utility as u
# import time
# from scipy.optimize import basinhopping
# from scipy.optimize import differential_evolution
# from skopt import gp_minimize
#
# X = np.array([[0,0,0,0,0,0]])
# Y = np.array([0])
# def objective(x):
#     global X, Y
#     y = funObj.objective_6D(x)
#     X = np.concatenate((X, np.array(x).reshape(1,6)))
#     Y = np.concatenate((Y, np.array([y])))
#     return y
#
# test_dir = r'./Airfoil6D_data/opt_GA/logs/'
# u.make_sure_path_exists(test_dir)
# funObj = f.Airfoil(3, 0.2, 5e005, test_dir, graphics)
# bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
#
# start_time = time.time()
#
# ret = differential_evolution(objective, bounds)
#
# np.savez(r'./Airfoil6D_data/opt_GA/Airfoil6D_XY_GA.npz',np.array(res.x_iters), res.func_vals, time.time() - start_time)
# #xy  = np.load(r'./Airfoil6D_data/opt_SKOPT/Airfoil6D_XY_SKOPT.npz')
# print('run time: ', time.time() - start_time)



# ################# convergence plots #############################


import matplotlib.pyplot as plt
import plot_convergence as conv
import  numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#############   SKOPT   #################################
alpha = 0.25
consider = 200
fontsize=13

number = 5
sum = np.zeros(consider)
for i in range(number):
    xy_gp = np.load(r'./Airfoil6D_data/opt_SKOPT_{}/Airfoil6D_XY_SKOPT.npz'.format(i))
    y_gp = xy_gp['arr_1']
    y_gp = conv.sort(y_gp)
    plt.plot(y_gp[0:consider], 'r-', alpha=alpha)# label='GP based-{}'.format(i),
    sum = sum + np.array(y_gp[0:consider])
sum = sum/number
plt.legend()
plt.plot(sum, 'r-', label='SKOPT-EI(scikit-learn)')

number = 5
sum = np.zeros(consider)
for i in range(number):
    xy_gp = np.load(r'./Airfoil6D_data/opt_SKOPT_LB_{}/Airfoil6D_XY_SKOPT_LB.npz'.format(i))
    y_gp = xy_gp['arr_1']
    y_gp = conv.sort(y_gp)
    plt.plot(y_gp[0:consider], 'g-', alpha=alpha) #, label='GP-LB-{}'.format(i)
    sum = sum + np.array(y_gp[0:consider])
sum = sum/number
plt.plot(sum, 'g-', label='SKOPT-LB(scikit-learn)')
plt.legend()
############   ANN      #################################
number = 5
sum = np.zeros(consider)
for i in range(number):
    xy_surrogate = np.load(r'./Airfoil6D_data/opt_ANN_distance_LB_{}/Airfoil_XY_ANN_distance_LB_itr_0.npz'.format(i))
    y_surrogate = xy_surrogate['arr_1']
    y_surrogate = conv.sort(y_surrogate)
    plt.plot(y_surrogate[0:consider], 'b-',  alpha = alpha) #label='ANN based-{}'.format(i),
    plt.legend()
    sum = sum + np.array(y_surrogate[0:consider])
sum = sum/number
plt.plot(sum, 'b-', label='N-SBAO')


# GA expeiment
xy_GA  = np.load(r'./Airfoil6D_data/opt_GA/Airfoil6D_XY_GA.npz')
x_GA = xy_GA['arr_0']
y_GA = xy_GA['arr_1']
y_GA = conv.sort(y_GA[0:5400:27])
plt.plot(y_GA, 'g-', label='GA')

i = 5
xy_GA  = np.load(r'./Airfoil6D_data/opt_GA/Airfoil6D_XY_GA.npz')
x_GA = xy_GA['arr_0'][0:3200:16]
y_GA = xy_GA['arr_1'][0:3200:16]
np.savez(r'./Airfoil6D_data/opt_ANN_distance_LB_{}/Airfoil_XY_ANN_distance_LB_itr_0'.format(i), x_GA, y_GA)

############  Random     #################################
number = 3
sum = np.zeros(consider)
for i in range(number):
    xy_random = np.load(r'./Airfoil6D_data/opt_RANDOM_{}/Airfoil6D_XY_RANDOM.npz'.format(i))
    y_random = xy_random['arr_1']
    y_random = conv.sort(y_random)
    plt.plot(y_random, 'k-',alpha = alpha) #  label='random search-{}'.format(i)
    plt.legend()
    sum = sum + np.array(y_random[0:consider])
sum = sum/number
plt.plot(sum, 'k-', label='Random Search')
plt.legend()
plt.xlabel('number of samples(n)')
plt.ylabel('neg. L/D')
plt.title('airfoil shape optmization in 6D')
############  GA     #################################
xy_GA  = np.load(r'./Airfoil6D_data/opt_GA/Airfoil6D_XY_GA.npz')
x_GA = xy_GA['arr_0']
y_GA = xy_GA['arr_1']
y_GA = conv.sort(y_GA)
plt.plot(y_GA, 'b-', label='random search')

plt.legend()
plt.title('convergence plot for 6D Airfoil shape optmization')
plt.xlabel('number of samples(n)')
plt.ylabel('minima after nth sample f_min(n)')
plt.annotate('initial samples = 100', xy=(50,y_gp[50]),xytext=(50,y_gp[50]+2))
plt.show()


for i in range(1,240,40):
    foil = np.loadtxt('./AirfoilGen/s0/context0/{}.dat'.format(i))
    plt.plot(foil[:, 0], foil[:, 1])
plt.show()


foil = np.loadtxt('./functions/test_DIR_/1.dat')





##################### generalisation ##################################
import matplotlib.pyplot as plt
import plot_convergence as conv
import  numpy as np

i = 4
j = 2
series = np.load(r'./AirfoilGen{}/series_0.npz'.format(i))
series = series['arr_0']

#xy_surrogate = np.load(r'./Airfoil6D_data/opt_ANN_distance_LB_0/Airfoil_XY_ANN_distance_LB_itr_0.npz')
xy_surrogate = np.load(r'./AirfoilGen{}/Airfoil_XY_series_0_context_{}.npz'.format(i, j))
x_surrogate = xy_surrogate['arr_0']
y_surrogate = xy_surrogate['arr_1']
y_surrogate = conv.sort(y_surrogate)
plt.plot(y_surrogate, label='ANN based-{}'.format(j))
plt.legend()





import plot_convergence as conv
import numpy as np
import matplotlib.pyplot as plt

class IsConverged:
    def __init__(self, max_patience=None):
        print("convergence objected created")
        self.patience = 0
        if max_patience == None:
            self.max_patience = 3
        else:
            self.max_patience = max_patience

    def isConverged(self, value):
        if value == True:
            self.patience = self.patience+1
            print('one Hit, patience = ', self.patience)
        elif self.patience > 0 and value == False:
            self.patience = self.patience-1
            print('one missed, patience = ', self.patience)
        else:
            print('way to go, patience = ', self.patience)
        if self.patience >= self.max_patience:
            return True
        else:
            return False



def get_count(track, target, tol, max_patience):
    conv_obj = IsConverged(max_patience)
    length = track.__len__()
    for i in range(length):
        isTrue = np.abs(track[i]-target) < tol
        isConverged = conv_obj.isConverged(isTrue)
        if isConverged == True:
            return i
    return length


number_itrs = 9
counts = np.zeros((number_itrs, 5))
for i in range(number_itrs):
    series = np.load('./AirfoilGen4-{}/series_0.npz'.format(i))
    #fig = plt.figure()
    for j in range(5):
        context = series['arr_0'][j]
        xy = np.load('./AirfoilGen4-{}/Airfoil_XY_series_0_context_{}.npz'.format( i,j))
        x = xy['arr_0']
        y = xy['arr_1']
        #funobj = FunObjectGenerator(context)
        #x_min_a, y_min_a = find_actual_minima(funobj)
        #y_min_a = -66
        #mask = x[:, 1] == context
        #xj = x[mask][:, 0]
        #yj = y[mask]
        x_min = x[y.argmin()]
        y_min = y.min()
        print('after context {}'.format(j))
        print('actual minima, x = {}, y = {} \n found minima, x = {}, y = {}'.format(999, 999, x_min, y_min))
        y = conv.sort(y)
        plt.plot(y, label=j)
        print('xj')
        print(x)
        print('yj')
        print(y)
        counts[i][j] = get_count(track=y, target=y_min , tol=np.abs(y_min * 0.01), max_patience=40)
        #counts[i][j] = get_count(track=y, target=y_min , tol=np.abs(y_min * 0.10), max_patience=20)
        #counts[i][j] = get_count(track=y, target=y_min * 0.99, tol=np.abs(y_min * 0.05), max_patience=25)

    plt.legend()

print(counts)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(number_itrs):
    ax.plot(counts[i], label = i, alpha = 0.4)
ax.plot(counts.mean(axis=0),'k', label = 'mean')
#plt.ylim(0,200)
plt.xlabel('number of contexts seen before')
plt.ylabel('number of points to reach 99% of minima with 1% acc')
plt.xticks([0,1,2,3,4])
plt.title('Convergence: f_min value, repeat=25, tol=1%')
ax.legend()













