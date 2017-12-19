import numpy as np
from keras import metrics
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scipy import optimize
from keras import optimizers

from src import helper as h, ei


def optimiseN(funObj, x0, budget,dropout,
              strategy=None,
              uncertainty='distance',
              minimizer_kwargs=None
              ):

    objective = funObj.objective
    dim = len(x0)
    np.random.seed(123)
    model = setupANN(dim, dropout=dropout)
    X, Y = prepareInitialTrainingData(objective, x0)
    for i in range(budget):
        print('<================Infill number {}============>'.format(i))
        model.fit(X, Y, epochs=300, batch_size=5, verbose=0)
        model.evaluate()

        acquisition_function = ei.get_expected_imp
        params = X, Y, model, 'distance'

        # ret = differential_evolution(acquisition_function, [(-5, 10), (0, 15)], args=params, disp=False)
        # #print("Neg EI minimum: x = [%.4f, %.4f], EI(x0) = %.4f" % (ret.x[0], ret.x[1], ret.fun))
        #
        # next_x = ret.x
        # next_y = objective(next_x)

        rranges = (slice(-5, 10,0.15),slice(0, 15,0.15))
        ret = optimize.brute(acquisition_function, rranges, args=params, full_output=True, finish=None)

        #print(ret)
        next_x = ret[0]
        next_y = objective(next_x)

        print('next_x: ',next_x)
        print('nextY: ', next_y)

        next_x = next_x.reshape((1, 2))
        next_y = np.array([[next_y]])
        X = np.concatenate((X, next_x))
        Y = np.concatenate((Y, next_y))

        h.plot_progress(objective, acquisition_function,[-5,10],[0,15], 100, 100, X,Y, model, 'distance')



    result = (X, Y)
    return result


def setupANN(dim, dropout):

    model = Sequential()

    # making the model graph, Stacking layers is done by .add():

    model.add(Dense(units=100, input_dim=dim, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(units=100, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(units=100, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    # configure the model's learning process; loss and optimisation etc
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=[metrics.mae])
    return model


# def setupANN(dim, nlayers, units, activation, dropout, loss, optimizer):
#     model = Sequential()
#
#     # making the model graph, Stacking layers is done by .add():
#
#     model.add(Dense(units=units, input_dim=dim, activation=activation))
#     model.add(Dropout(dropout))
#     for i in range(nlayers - 1):
#         model.add(Dense(units=units, input_dim=dim, activation=activation))
#         model.add(Dropout(dropout))
#
#     model.add(Dense(units=1))
#     # configure the model's learning process; loss and optimisation etc
#     model.compile(loss=loss,
#                   optimizer=optimizer,
#                   metrics=['accuracy'])
#
#     return model

def prepareInitialTrainingData(objective, x0):
    X = x0.reshape((1,2))
    y0 = objective(x0)
    Y = np.array([[y0]])
    return X,Y