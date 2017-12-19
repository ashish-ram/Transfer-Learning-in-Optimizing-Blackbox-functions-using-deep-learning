import  numpy as np
import pylab
import plotly as py
import matplotlib.pyplot as plt
from plotly.graph_objs import *

def plot_progress(fun, acquisition_function, x1range, x2range, x1resolution, x2resolution,  *args):
    def prediction(data):
        return model.predict(np.array([[data[0], data[1]]]))

    X,Y, model, method = args
    #print(args)
    xx, yy = pylab.meshgrid(
        pylab.linspace(x1range[0], x1range[1], x1resolution),
        pylab.linspace(x2range[0], x2range[1], x2resolution))

    zz_true = np.zeros(xx.shape)
    zz_pred = np.zeros(xx.shape)
    zz_ac = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz_true[i, j] = fun([xx[i, j], yy[i, j]])
            zz_pred[i, j] = prediction(np.array([xx[i, j], yy[i, j]]))
            zz_ac[i, j] = acquisition_function(np.array([xx[i, j], yy[i, j]]),X,Y, model, 'distance')

    # plot the calculated function values
    fig = plt.figure(figsize=(15, 4))
    fig.add_subplot(1, 3, 1)
    plt.pcolor(xx, yy, zz_true)
    plt.plot(X[:, 0], X[:, 1], 'rx')
    plt.title('True Function')
    plt.colorbar()

    fig.add_subplot(1, 3, 2)
    plt.pcolor(xx, yy, zz_ac)
    plt.plot(X[:, 0], X[:, 1], 'rx')
    plt.title('negative EI')
    plt.colorbar()

    fig.add_subplot(1, 3, 3)
    plt.pcolor(xx, yy, zz_pred)
    plt.plot(X[:, 0], X[:, 1], 'rx')
    plt.title('Surrogate with points')
    plt.colorbar()



    plt.show()

    #return ax



def explore(fun, bounds, x1resolution, x2resolution, title, ax=None, points=None):
    x1range = bounds[0]
    x2range = bounds[1]
    if ax is None:
        ax=pylab.gca()
    xx, yy = pylab.meshgrid(
        pylab.linspace(x1range[0], x1range[1], x1resolution),
        pylab.linspace(x2range[0], x2range[1], x2resolution))

    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = fun([xx[i, j], yy[i, j]])

    # plot the calculated function values
    im = ax.pcolor(xx, yy, zz)
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], 'rx')
        #ax.colorbar(im)
        #ax.title(title)
    #pylab.show()
    #return ax

    #SURFACE PLOT
    py.offline.init_notebook_mode(connected=True)

    import pandas as pd

    data1 = Surface(x=xx, y=yy, z=zz)
    data = [data1]
    layout = Layout(
        title='{} Function '.format(str(fun)),
        autosize=True,  # False
        width=500,
        height=500,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = Figure(data=data, layout=layout)
    py.offline.iplot(fig, filename='elevations-3d-surface')
    return ax


def explore_cont(fun, bounds, x1resolution, x2resolution,V, ax=None, points=None):
    x1range = bounds[0]
    x2range = bounds[1]
    if ax is None:
        ax=pylab.gca()
    xx, yy = pylab.meshgrid(
        pylab.linspace(x1range[0], x1range[1], x1resolution),
        pylab.linspace(x2range[0], x2range[1], x2resolution))

    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = fun([xx[i, j], yy[i, j]])

    # plot the calculated function values
    #ax.pcolor(xx, yy, zz)
    im = ax.contourf(xx,yy,zz, V)
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], 'rx')

    return im



def explore_EI(fun, x1range, x2range, x1resolution, x2resolution, title,ax=None, *args):
    if ax is None:
        ax = pylab.gca()
    X,Y, model, method = args
    #print(args)
    xx, yy = pylab.meshgrid(
        pylab.linspace(x1range[0], x1range[1], x1resolution),
        pylab.linspace(x2range[0], x2range[1], x2resolution))

    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = fun(np.array([xx[i, j], yy[i, j]]), X,Y, model, method)

    # plot the calculated function values
    ax = ax.pcolor(xx, yy, zz)
    # if points is not None:
    #     pylab.plot(points[:, 0], points[:, 1], 'rx')
    #pylab.colorbar()
#    ax.title(title)
    #pylab.show()
    return ax

    # # SURFACE PLOT
    # py.offline.init_notebook_mode(connected=True)
    #
    # import pandas as pd
    #
    # data1 = Surface(x=xx, y=yy, z=zz)
    # if points is not None:
    #     data2 = Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
    #                       mode='markers',
    #                       marker=dict(
    #                           size=5,
    #                           line=dict(color='rgba(217, 217, 217, 0.94)', width=0.1),
    #                           opacity=1
    #                       ))
    #     data = [data1, data2]
    # else:
    #     data = [data1]
    # layout = Layout(
    #     title='{} Function '.format(str(fun)),
    #     autosize=True,  # False
    #     width=500,
    #     height=500,
    #     margin=dict(
    #         l=65,
    #         r=50,
    #         b=65,
    #         t=90
    #     )
    # )
    # fig = Figure(data=data, layout=layout)
    # py.offline.iplot(fig, filename='elevations-3d-surface')




if __name__=="__main__":
    def objective(x):
        x1 = x[0]
        x2 = x[1]  # self.context
        term1 = (4 - 2.1 * x2 * x2 + (np.power(x2, 4)) / 3) * np.power(x2, 2)
        term2 = x1 * x2
        term3 = (-4 + 4 * x1 * x1) * x1 * x1
        y = term1 + term2 + term3
        return y

    from functions import functions
    branin = functions.Branin()
    f, ax  = plt.subplots(1,1)
    V = np.linspace(-5, 320, 20)
    ax = explore_cont(branin.objective, branin.bounds, 100, 100,V,ax=ax)

    f, ax = plt.subplots(1, 1)
    V = np.linspace(-5, 5, 20)
    im = explore_cont(objective, ((-2,2),(-2,2)), 100, 100, V, ax=ax)
    f.colorbar(im, ax=ax)

    f, ax = plt.subplots(1, 1)
    V = np.linspace(-3, 10, 20)
    im = explore(objective, ((-1, 1), (-2, 2)), 100, 100, title='title',ax=ax)

