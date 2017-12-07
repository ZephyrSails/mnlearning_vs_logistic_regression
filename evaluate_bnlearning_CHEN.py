import numpy as np

import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import linear_model, datasets

WORDS   = 9314
DOC     = 153
Spam    = 28
Ham     = 125


def read_data(k):
    X = [[] for _ in xrange(k)]
    Y = [[] for _ in xrange(k)]
    with open('data2.txt', 'r') as datafile:
        for i, line in enumerate(datafile.readlines()):
            line = line.split(' ')
            x = np.array([0. for _ in xrange(WORDS)])
            y = float(line[-1])
            for word in line[:-1]:
                x[int(word)] = 1.0

            X[i % k].append(x)
            Y[i % k].append(y)

    return X, Y


def score(X, Y, theta):
    n = len(X[0])
    dummy = [0.0 for _ in xrange(n)]
    predict = np.array([0.0 for _ in xrange(len(Y))])

    for i, x in enumerate(X):
        dY_1 = np.array(list(x) + dummy + [1.])
        dY_0 = np.array(dummy + list(x) + [0.])

        predict[i] = 1.0 if (sum(dY_1 * theta) < sum(dY_0 * theta)) else 0.0

    correct = 0.

    # predict = np.array([0.0 for _ in xrange(len(Y))])

    for i, y in enumerate(Y):
        if predict[i] == y:
            correct += 1.

    print correct / len(Y)
    print predict, Y


if __name__ == '__main__':
    X, Y = read_data(1)
    theta = np.load('theta_2.14.npy')

    # logreg.fit(X[0], Y[0])
    print score(X[0], Y[0], theta)
    # 0.888524590164

    # logreg = linear_model.LogisticRegression(C=1e5)
    # logreg.fit(X[1], Y[1])
    # print logreg.score(X[0], Y[0])
