import matplotlib.pyplot as plt
import numpy as np

def sort(Y):
    """
    
    :param Y: numpy array with all the Y values of the trajctory
    :return: sorted list with fun minima at that iteration
    """
    min_track = []
    for i in range(len(Y)):
        if i == 0:
            min_track.append(Y[i])
        elif Y[i] > min_track[i - 1]:
            min_track.append(min_track[i - 1])
        else:
            min_track.append(Y[i])
    return min_track

# YY = np.load("YY_ANN_RBF.npz")
# YY = YY['arr_0']

def plot_mean_convergence(YY):
    """
    
    :param YY: a list of numpy arrays(samplesx1) with each array trajectory of algo optimisation 
    :return: 
    """

    for i in range(len(YY)):
        yy = YY[i]
        yy = sort(yy)
        plt.plot(yy, ".-", label=str(i), alpha=1)
    mean_y = YY.mean(axis=0)
    plt.plot(mean_y, '.-', label='mean line')
    plt.title('Optimisation Convergence')
    plt.xlabel('min f(x) after n calls')
    plt.ylabel('number of funciton calls (n)')
    plt.legend()
    plt.show()
