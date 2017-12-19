import numpy as np
import keras
from keras import optimizers
from keras import backend as K
from keras.layers import Dense, Dropout, Activation
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from pyDOE import *
from sklearn.model_selection import PredefinedSplit
import time
from src import sampling
from functions import functions

start = time.time()


funObj = functions.Branin()

def create_model(layers = 2, units = 40, eps=1e-8, lr = 1e-5):
    model = Sequential()
    dim= 2
    ## making the model graph, Stacking layers is done by .add():
    model.add(Dense(units=units, input_dim=dim, activation='sigmoid'))
    for i in range(layers-1):
        model.add(Dense(units=units, activation='sigmoid'))

    model.add(Dense(units=1))

    # optmiser = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=eps, decay=0.0)

    # configure the model's learning process; loss and optimisation etc
    model.compile(loss='mse',
                  optimizer=optimizer, metrics=["mae"])
    print('model created with layer= {}, units= {}, eps={}, lr={}'.format(layers, units, eps, lr))
    return model

## Generate 6D data:
dim = 2
samples = 30
X, Y = sampling.sample_training_data(funObj.objective, dim=2, samples=samples, bounds=funObj.bounds)

##### making sure that the training and validation sets are the same during cross validation of gridsearchCV
XX = np.concatenate((X, X))
YY = np.concatenate((Y, Y))
YY = YY.flatten()
train_fold =[0]*samples
valid_fold = [-1]*samples
test_fold = np.array(train_fold+valid_fold)
ps = PredefinedSplit(test_fold)
print(XX)
print(YY)
print(ps.test_fold)

# # fix the seed
# seed = 7
# numpy.random.seed(seed)


# create model
model = KerasRegressor(build_fn=create_model, verbose=2)
# define the grid search parameters
batch_size = [5]
epochs = [10000]
units = [40,100,200]
layers = [2,3]
eps = [1e-8, 1e-4, 1e-2, 1e-1, 1]
lr = [1e-5, 1e-3, 1e-2, 0.1]
#optimizer = ['SGD', 'RMSprop','Adam']# 'Adagrad', 'Adadelta', ]
param_grid = dict(batch_size=batch_size, epochs=epochs, layers=layers, units=units,eps=eps, lr=lr)
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv = ps,
                    n_jobs=-1)
grid_result = grid.fit(XX, YY)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

end = time.time()
print("======================================================")
print('time takesn = {}'.format(end - start))