import numpy as np

# import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

WORDS   = 9314
DOC     = 305
Spam    = 56
Ham     = 249


def read_data(k):
    X = [[] for _ in xrange(k)]
    Y = [[] for _ in xrange(k)]
    with open('data.txt', 'r') as datafile:
        for i, line in enumerate(datafile.readlines()):
            line = line.split(' ')
            x = np.array([0. for _ in xrange(WORDS)])
            y = float(line[-1])
            for word in line[:-1]:
                x[int(word)] = 1.0

            X[i % k].append(x)
            Y[i % k].append(y)

    return X, Y

if __name__ == '__main__':
    X, Y = read_data(2)
    logreg = linear_model.LogisticRegression(C=1e5)

    logreg.fit(X[0], Y[0])
    print logreg.score(X[1], Y[1])

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X[1], Y[1])
    print logreg.score(X[0], Y[0])
