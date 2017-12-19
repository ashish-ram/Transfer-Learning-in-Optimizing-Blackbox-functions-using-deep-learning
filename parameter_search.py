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
import time

"""
This param search uses GridsearchCV for finding optimum 
1. batch size for training
2. number of epochs 
3. optimiser type
4. number of neurons per layer in 3 layered network
"""
start = time.time()
def fun1D(x):
    return np.power(6.0*x-2.0,2.0)*np.sin(12.0*x-4.0)
X = np.array([[ 0.40326804],
        [ 0.20383569],
        [ 0.08074295],
        [ 0.81216408],
        [ 0.84788367],
        [ 0.16199875],
        [ 0.92205915],
        [ 0.56114674],
        [ 0.36259569],
        [ 0.69901181],
        [ 0.53908923]])
# X = np.array([[ 0.40326804],
#         [ 0.69901181],
#         [ 0.53908923]])
# X = np.linspace(0,1,50).reshape(50,1)
Y = fun1D(X)

# from sklearn.model_selection import PredefinedSplit
# X = np.array([[0.1], [0.2], [0.4], [0.8], [0.1], [0.2], [0.4], [0.8]])
# Y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
# test_fold = [0,0,0,0,-1,-1,-1,-1]
# ps = PredefinedSplit(test_fold)

# from sklearn.model_selection import PredefinedSplit
# X = np.array([[0.1], [0.2], [0.4], [0.8], [0.13], [0.26], [0.44], [0.87],[0.31], [0.92], [0.84],  #=== >
#               [0.1], [0.2], [0.4], [0.8], [0.13], [0.26], [0.44], [0.87],[0.31], [0.92], [0.84]])
# Y = np.array([1, 2, 1, 1, 0, 0, 1, 1, 3, 4, 7,
#               1, 2, 1, 1, 0, 0, 1, 1, 3, 4, 7])
# test_fold = [0,0,0,0,0,0,0,0,0,0,0,
#             -1,-1,-1,-1,-1,-1,-1,-1, -1, -1, -1]
# ps = PredefinedSplit(test_fold)

##### making sure that the training and validation sets are the same during cross validation of gridsearchCV
from sklearn.model_selection import PredefinedSplit
XX = np.concatenate((X, X))
YY = np.concatenate((Y, Y))
YY = YY.flatten()
samples = 11
train_fold =[0]*samples
valid_fold = [-1]*samples
test_fold = np.array(train_fold+valid_fold)
ps = PredefinedSplit(test_fold)

def create_model(optimizer='adam', units=40):
    print('model created')
    model = Sequential()
    #units = 100
    dim= 1
    ## making the model graph, Stacking layers is done by .add():
    model.add(Dense(units=units, input_dim=dim, activation='sigmoid'))
    model.add(Dense(units=units, activation='sigmoid'))
    model.add(Dense(units=units, activation="sigmoid"))
    model.add(Dense(units=1))

    # optmiser = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optmiser = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # configure the model's learning process; loss and optimisation etc
    model.compile(loss='mse',
                  optimizer=optimizer, metrics=["mae"])

    return model

seed = 7
numpy.random.seed(seed)


# create model
model = KerasRegressor(build_fn=create_model, verbose=2)
# define the grid search parameters
units = [40]#,60,100,200]
batch_size = [5]#, 10]
epochs = [6000]#, 1000, 5000, 10000, 15000]
optimizer = ['Adam']#, 'RMSprop', 'Adam', ] #'Adagrad', 'Adadelta', 'Adamax', 'Nadam'
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, units=units)
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv = ps,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1
                    )#,scoring='r2')#, )scoring='mean_absolute_error'n_jobs=-1



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
print('time takesn = {}'.format(end-start))