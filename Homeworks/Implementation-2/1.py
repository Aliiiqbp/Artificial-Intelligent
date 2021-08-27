
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import time


def data_generator(start, end, num, funct):
    data = pd.DataFrame(
        np.concatenate((np.random.uniform(start, end, (num, 2)), -np.ones((num, 1))), axis=1),
        columns=['x', 'y', 'Class'])
    data.Class[funct(data.x, data.y)] = 1
    return data


def plot_separator(svc):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svc.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100, linewidth=1,
               facecolors='none', edgecolors='k')


def show(data, svc):
    plt.figure()
    plt.scatter(data.x, data.y, c=data.Class, s=100)
    plot_separator(svc)
    plt.show()


start_point, end_point = -5, 5
size = 100
cnt = [100 * i for i in range(1, 101)]
tim = []
for i in range(100):
    start_time = time.time()
    dataset = data_generator(start_point, end_point, size, lambda x, y: x ** 3 - 3 * x < y ** 2)
    model = SVC(kernel='poly')
    model.fit(dataset[['x', 'y']], dataset.Class)
    print("--- %s seconds ---" % (time.time() - start_time))
    show(dataset, model)
    tim.append(time.time() - start_time)
    size += 100

plt.figure()
plt.plot(cnt, tim, 'x')
plt.xlabel("number of nodes")
plt.ylabel("time")
plt.show()
