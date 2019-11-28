import pandas as pd
import numpy as np


def h(x, w, b):
    return 1 / (1+np.exp(-(np.dot(x, w) + b)))


if __name__ == '__main__':
    dataset = pd.read_csv('voice.csv', usecols=[*range(0, 20)])
    lables = pd.read_csv('voice.csv', usecols=[20])
    data_x = dataset.to_numpy()
    lablesNP = lables.to_numpy().flatten()
    lablesArr = []
    for index, item in  enumerate(lablesNP):
        if item == 'male':
            lablesArr.append(1)
        else:
            lablesArr.append(0)
    w = np.zeros(20)

    data_y = np.array(lablesArr)
    b = 0
    alpha = 0.001
    for iteration in range(100000):
        gradient_b = np.mean(1 * ((h(data_x, w, b)) - data_y))
        gradient_w = np.dot((h(data_x, w, b) - data_y), data_x) * 1 / len(data_y)
        b -= alpha * gradient_b
        w -= alpha * gradient_w

    print("user number 10 is probaly: ", h(data_x[10], w, b))
    print("user number 1000 is probaly: ", h(data_x[1000], w, b))
    print("user number 3000 is probaly: ", h(data_x[3000], w, b))
    print("user number 2555 is probaly: ", h(data_x[2555], w, b))
