import numpy.random as random
import pylab
from functions import functions as f
from src import sampling
from src import Acquisition
import keras
from keras.models import Sequential
from keras.layers import Dense
from pyDOE import *
import airfoil.utility as u



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
    objective = funObj.objective_opt
    bounds = funObj.bounds
    x1range = bounds[0]
    x2range = bounds[1]
    dim = 2
    X, Y = sampling.sample_training_data(objective, dim=2,
                                         samples=initial_samples, bounds=funObj.bounds)
    # early_stopping = EarlyStopping(monitor='loss',min_delta=0, patience=100, verbose=1)
    print("Initial Samples: ")
    print("X: ", X)
    print("Y: ", Y)
    for i in range(50):  # number of infills
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
        next_y = np.array([funObj.objective_opt(next_x)])
        next_x = next_x.reshape(1, 2)
        next_y = next_y.reshape(1, 1)
        X = np.concatenate((X, next_x))
        Y = np.concatenate((Y, next_y))

        zz_pred = model.predict(x)
        #plot(funObj, X, hist, zz_pred, LB, s)
        ##### lool over ########

    print('minima found is f(x) = {}'.format(Y.min()))
    return model, X, Y

test_dir = r'./Airfoil_test/opt/logs/'
u.make_sure_path_exists(test_dir)
funObj = f.Airfoil(3, 0.2, 5e005, test_dir, True)

for i in range(1):
    initial_samples = 15 + random.randint(3)
    print('Starting iteration number : ', i)
    print('initial number of samples: ', initial_samples)
    model, X, Y = single_run(funObj, initial_samples=initial_samples)
    model.save(r"./Airfoil_test/Airfoil_ANN_distance_LB_itr_{}".format(i))
    np.savez(r'./Airfoil_test/Airfoil_XY_ANN_distance_LB_itr_{}'.format(i),X,Y)


################## visualise ground truth ###############################
#
# import numpy as np
# import pylab
# import matplotlib.pyplot as plt
# import plotly as py
# from plotly.graph_objs import *
#
# #### load the data ##############
# path = r'./Airfoil_data/grids/2D/'
# lds = np.loadtxt(path+'/grid_labels.dat')
# grid = np.loadtxt(path+'/grid_data/grid.dat')
#
# ##### find the maxima points####
# max_ld_at = np.argmax(lds)
#
# #####reshaping the data for colormap or surfaceplots #####
# y11, y12, y21, y22 = -1, 1, -1, 1
# resolution = 100
# lds_reshaped = lds.reshape((resolution, resolution))
# lds_reshaped = lds_reshaped.transpose()
#
# ##### Plot heatmap ############
# plt.imshow(lds_reshaped, cmap='hot', interpolation=None, origin='lower', extent=[y11, y12, y21, y22])
#
# ##### plot maximas ############
# plt.plot(grid[max_ld_at,1], grid[max_ld_at, 2], 'bx')
#
#
# plt.colorbar()
# plt.title("Ground Truth of grid 100x100")
# print('LD max at: ', grid[max_ld_at], 'value: ', lds[max_ld_at])
# plt.show()
#
# #py.offline.init_notebook_mode(connected=True)
# y11, y12, y21, y22 = -1, 1, -1, 1
# resolution = 100
# xx, yy = pylab.meshgrid(
#     pylab.linspace(y11,y12,resolution),
#     pylab.linspace(y21,y22,resolution))
#
# surface = Surface(x=xx, y=yy, z=lds_reshaped)
#
# data = Data([surface])
#
# layout = Layout(
#     title='Airfoil input space zoomed near the maxima',
#     autosize=False,
#     width=700,
#     height=700,
#     margin=dict(
#         l=65,
#         r=50,
#         b=65,
#         t=90
#     )
# )
#
# fig = Figure(data=data, layout=layout)
# py.offline.iplot(fig, filename='parametric_plot.html', image='png')

############  plots ################################

import matplotlib.pyplot as plt
import plot_convergence as conv
import  numpy as np

#############   SKOPT   #################################
alpha = 0.25
consider = 50

number = 2
sum = np.zeros(consider)
for i in range(number):
    xy_gp = np.load(r'./Airfoil_data/SKOPT/Airfoil_XY_SKOPT_itr_{}.npz'.format(i))
    y_gp = xy_gp['arr_1']
    y_gp = conv.sort(y_gp)
    plt.plot(y_gp[0:consider], 'r-', alpha=alpha)# label='GP based-{}'.format(i),
    sum = sum + np.array(y_gp[0:consider])
sum = sum/number
plt.plot(sum, 'r-', label='GP-EI-mean')
# plt.legend()
#
# number = 2
# sum = np.zeros(consider)
# for i in range(number):
#     xy_gp = np.load(r'./Airfoil6D_data/opt_SKOPT_LB_{}/Airfoil6D_XY_SKOPT_LB.npz'.format(i))
#     y_gp = xy_gp['arr_1']
#     y_gp = conv.sort(y_gp)
#     plt.plot(y_gp[0:consider], 'g-', alpha=alpha) #, label='GP-LB-{}'.format(i)
#     sum = sum + np.array(y_gp[0:consider])
# sum = sum/number
# plt.plot(sum, 'g-', label='GP-LB-mean')
# # plt.legend()
############   ANN      #################################
number = 2
sum = np.zeros(consider)
for i in range(number):
    xy_surrogate = np.load(r'./Airfoil_data/Airfoil_XY_ANN_distance_LB_itr_{}.npz'.format(i))
    y_surrogate = xy_surrogate['arr_1']
    y_surrogate = conv.sort(y_surrogate)
    plt.plot(y_surrogate[0:consider], 'b-',  alpha = alpha) #label='ANN based-{}'.format(i),
    sum = sum + np.array(y_surrogate[0:consider])
sum = sum/number
plt.plot(sum[:,0], 'b-', label='ANN-LB-mean')

plt.legend()
plt.xlabel('number of samples(n)')
plt.ylabel('neg. L/D')
plt.title('airfoil shape optmization in 2D')