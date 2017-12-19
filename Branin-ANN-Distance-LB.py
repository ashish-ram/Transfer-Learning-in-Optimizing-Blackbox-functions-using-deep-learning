import numpy.random as random
import pylab
from functions import functions as f
from src import sampling
from src import Acquisition
import keras
from keras.models import Sequential
from keras.layers import Dense
from pyDOE import *

def setupANN(dim, units, loss):
    model = Sequential()

    ## making the model graph, Stacking layers is done by .add():
    model.add(Dense(units=units, input_dim=dim, activation='sigmoid'))
    model.add(Dense(units=units, activation='sigmoid'))
    model.add(Dense(units=units, activation="sigmoid"))
    model.add(Dense(units=1, activation='linear'))

    # optmiser = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optmiser = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # configure the model's learning process; loss and optimisation etc
    model.compile(loss=loss,
                  optimizer=optmiser, metrics=["mae"])

    return model


def single_run(funObj, initial_samples):
    objective = funObj.objective
    bounds = funObj.bounds
    x1range = bounds[0]
    x2range = bounds[1]
    dim = 2
    X, Y = sampling.sample_training_data(objective, dim=2,
                                         samples=initial_samples, bounds=funObj.bounds)
    # early_stopping = EarlyStopping(monitor='loss',min_delta=0, patience=100, verbose=1)

    for i in range(20):  # number of infills
        print('==========> infill {}'.format(i))
        model = setupANN(dim=2, units=200, loss='mean_squared_error')
        hist = model.fit(X, Y, epochs=8000, batch_size=5, verbose=0)

        xx, yy = pylab.meshgrid(
            pylab.linspace(x1range[0], x1range[1], 100),
            pylab.linspace(x2range[0], x2range[1], 100))
        x = []
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x.append([xx[i, j], yy[i, j]])
        x = np.array(x)
        EIobj = Acquisition.Acquisition(X=X, y=Y, model=model)
        LB, s = EIobj.get_lower_bound_distance(x, 0.92)  # A = 0.95 (balanced), A = 0.92(exploitative)

        next_x = x[LB.argmin()]
        next_y = np.array([funObj.objective(next_x)])
        next_x = next_x.reshape(1, 2)
        next_y = next_y.reshape(1, 1)
        X = np.concatenate((X, next_x))
        Y = np.concatenate((Y, next_y))

        zz_pred = model.predict(x)
        #plot(funObj, X, hist, zz_pred, LB, s)
        ##### lool over ########

    print('minima found is f(x) = {}'.format(Y.min()))
    return model, X, Y


funObj = f.Branin()

for i in range(15):
    initial_samples = 5 + random.randint(3)
    print('Starting iteration number : ', i)
    print('initial number of samples: ', initial_samples)
    model, X, Y = single_run(funObj, initial_samples=initial_samples)
    model.save(r"./Branin_data/Branin_ANN_distance_LB_itr_{}".format(i))
    np.savez(r'./Branin_data/Branin_XY_ANN_distance_LB_itr_{}'.format(i),X,Y)

####### plot branin function #########################

import matplotlib.pylab as plt
from functions import functions as f
from keras.models import load_model
import pylab


funObj = f.Branin()
bounds = funObj.bounds
x1range = bounds[0]
x2range = bounds[1]
xx, yy = pylab.meshgrid(
            pylab.linspace(x1range[0],x1range[1], 100),
            pylab.linspace(x2range[0],x2range[1], 100))
x = []
zz = []
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x.append([xx[i, j], yy[i, j]])
        zz.append(funObj.objective([xx[i, j], yy[i, j]]))
x = np.array(x)
zz = np.array(zz)
fig = plt.figure(figsize=(10,10))
V = np.linspace(-5, 320, 20)
plt.contourf(xx, yy, zz.reshape(100, 100), V)
plt.savefig('./Figures/branin.eps')
plt.xlabel('x1', fontsize = fontsize)
plt.ylabel('x2', fontsize = fontsize)

################### for Plotting all of them #######################
import plot_convergence
import matplotlib.pylab as plt
fig = plt.figure(figsize=(20,10))
fig.suptitle("Branin 15 iterations with ANN Distance LB")
for i in range(3):
    for j in range(5):
        xy = np.load(r'./Branin_data/Branin_XY_ANN_distance_LB_itr_{}.npz'.format(i*5+j))
        x = xy['arr_0']
        y = xy['arr_1']
        # print(x[y.argmin()])
        # print(y.argmin())
        y_sorted = plot_convergence.sort(y)
        fig.add_subplot(3,5,i*5+j+1)
        #plt.plot(y, 'b--', label = "infill points", alpha =0.35)
        plt.plot(y_sorted, 'r.-', label='iternation#{}'.format(i*5+j))
        plt.title('x_min={0}, fmin={1:.2f}'.format(x[y.argmin()], y.min()),fontsize=8)
        plt.legend()

################# plot average of all ##############################
import plot_convergence
import matplotlib.pylab as plt
import numpy as np
fig = plt.figure(figsize=(20,10))
fig.suptitle("Branin 15 iterations with ANN Distance LB-Mean line convergence")
fig.add_subplot(1,1,1)
y_mean = np.zeros((27,1)) #27 is the maximun number os steps-->determinded manually
for i in range(15):
    xy = np.load(r'./Branin_data/Branin_XY_ANN_distance_LB_itr_{}.npz'.format(i))
    x = xy['arr_0']
    y = xy['arr_1']
    y = np.array(plot_convergence.sort(y))
    plt.plot(y,'b-' ,alpha = .25) #label='itr {}'.format(i),
    for k in range(y.shape[0]):
        y_mean[k] = (y_mean[k]*i + y[k])/(i+1)
plt.title('average min = {}'.format(y_mean.min()))
plt.plot(y_mean, 'b.-',label="N-SBAO-mean")
plt.legend()
################## visualize the model predictions ######################
import matplotlib.pylab as plt
from functions import functions as f
from keras.models import load_model
import pylab


funObj = f.Branin()
bounds = funObj.bounds
x1range = bounds[0]
x2range = bounds[1]
xx, yy = pylab.meshgrid(
            pylab.linspace(x1range[0],x1range[1], 100),
            pylab.linspace(x2range[0],x2range[1], 100))
x = []
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x.append([xx[i, j], yy[i, j]])
x = np.array(x)
fig = plt.figure(figsize=(20,10))
V = np.linspace(-5, 320, 20)
fig.suptitle("Model predictions")

for i in range(3):
    for j in range(5):
        model = load_model(r"./Branin_data/Branin_ANN_distance_LB_itr_{}".format(i*5+j))
        zz_pred = model.predict(x)
        fig.add_subplot(3, 5, i * 5 + j + 1)
        plt.contourf(xx, yy, zz_pred.reshape(100, 100), V)
        # plt.pcolor(xx, yy , zz_pred.reshape(100,100))
        plt.title('predictions')
        plt.colorbar()


######################################################     Plots for SKOPT  #############################################################
from skopt import gp_minimize
import numpy as np
import matplotlib.pyplot as plt
from functions import functions as f
import numpy.random as random

funObj = f.Branin()
def objective(x):
    return funObj.objective(x)
itr_num = 15
for i in range(itr_num):
    initial_samples = 5 + random.randint(3)
    print('Starting iteration number : ', i)
    print('initial number of samples: ', initial_samples)
    res = gp_minimize(objective,  # the function to minimize
                      [(-5.0, 10.0), (0.0, 15.0)],  # the bounds on each dimension of x
                      acq_func="LCB",  # the acquisition function
                      n_calls=25,  # the number of evaluations of f
                      n_random_starts=initial_samples,  # the number of random initialization points
                      # noise=0.1**2,       # the noise level (optional)
                      random_state=random.randint(500)
                      )  # the random seed
    X = np.array(res.x_iters)
    Y = np.array(res.func_vals)
    np.savez(r'./Branin_data/Branin_XY_SKOPT_itr_{}'.format(i), X, Y)

################### for Plotting all of them #######################
import plot_convergence
import matplotlib.pylab as plt
fig = plt.figure(figsize=(20,10))
fig.suptitle("Branin 15 iterations with SKOPT LCB")
for i in range(3):
    for j in range(5):
        xy = np.load(r'./Branin_data/Branin_XY_SKOPT_itr_{}.npz'.format(i*5+j))
        x = xy['arr_0']
        y = xy['arr_1']
        # print(x[y.argmin()])
        # print(y.argmin())
        y_sorted = plot_convergence.sort(y)
        fig.add_subplot(3,5,i*5+j+1)
        #plt.plot(y, 'b--', label = "infill points", alpha =0.35)
        plt.plot(y_sorted, 'r.-', label='iternation#{}'.format(i*5+j))
        plt.title('x_min={0}, fmin={1:.2f}'.format(x[y.argmin()], y.min()),fontsize=8)
        plt.legend()


################# plot average of all ##############################

import plot_convergence
import matplotlib.pylab as plt
fig = plt.figure(figsize=(20,10))
fig.suptitle("Branin 15 iterations with ANN Distance LB-Mean line convergence")
fig.add_subplot(1,1,1)
y_mean = np.zeros((25,1)) #27 is the maximun number os steps-->determinded manually
for i in range(15):
    xy = np.load(r'./Branin_data/Branin_XY_SKOPT_itr_{}.npz'.format(i))
    x = xy['arr_0']
    y = xy['arr_1']
    y = np.array(plot_convergence.sort(y))
    plt.plot(y,'r-',  alpha = .25) #label='itr {}'.format(i),
    for k in range(y.shape[0]):
        y_mean[k] = (y_mean[k]*i + y[k])/(i+1)

plt.title('average min = {}'.format(y_mean.min()))
plt.plot(y_mean, 'r.-',label="mean")
plt.legend()


################## skopt vs N-SBAO ###############################################
import plot_convergence
import matplotlib.pylab as plt
import numpy as np

fig = plt.figure(figsize=(20,10))
fig.suptitle("SKOPT vs N_SBAO for branin 2D, 15 iterations")
fig.add_subplot(1,1,1)
y_mean1 = np.zeros((25,1)) #27 is the maximun number os steps-->determinded manually
for i in range(15):
    xy = np.load(r'./Branin_data/Branin_XY_SKOPT_itr_{}.npz'.format(i))
    x = xy['arr_0']
    y = xy['arr_1']
    y = np.array(plot_convergence.sort(y))
    plt.plot(y,'r-',  alpha = .25) #label='itr {}'.format(i),
    for k in range(y.shape[0]):
        y_mean1[k] = (y_mean1[k]*i + y[k])/(i+1)

plt.plot(y_mean1, 'r.-',label="SKOPT-mean")


y_mean2 = np.zeros((27,1)) #27 is the maximun number os steps-->determinded manually
for i in range(15):
    xy = np.load(r'./Branin_data/Branin_XY_ANN_distance_LB_itr_{}.npz'.format(i))
    x = xy['arr_0']
    y = xy['arr_1']
    y = np.array(plot_convergence.sort(y))
    plt.plot(y,'b-' ,alpha = .25) #label='itr {}'.format(i),
    for k in range(y.shape[0]):
        y_mean2[k] = (y_mean2[k]*i + y[k])/(i+1)

plt.plot(y_mean2, 'b.-',label="N-SBAO-mean")
plt.xlabel('number of samples(n)')
plt.ylabel('minima after n samples')
plt.title('actual y_min = {} \n av y_min(SKOPT) = {} \n av y_min(N-SBAO) = {}'.format(0.397887,y_mean1.min(),y_mean2.min()))
plt.legend()
