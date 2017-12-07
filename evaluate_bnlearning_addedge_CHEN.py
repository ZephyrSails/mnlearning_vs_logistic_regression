import numpy as np

import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import linear_model, datasets

WORDS   = 9314
DOC     = 153
Spam    = 28
Ham     = 125
WPN = 30


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

def read_wordpair(wpn):
    wordpair = []
    with open('PMI_smothed_100.txt', 'r') as datafile:
        for line in  datafile.readlines():
            line = line.split(', ')
            
            if line[0]!=line[1]:
                wordpair.append([int(line[0]), int(line[1])])
            
    return wordpair[:wpn]


def segmentfile():
    datanum1_spam = Spam / 2
    datanum1_ham = Ham / 2

    count_spam = 0
    count_ham = 0
    file1 = open("data1.txt", 'w')
    file2 = open("data2.txt", 'w')
    with open('data.txt', 'r') as datafile:
        for line in  datafile.readlines():
            linelst = line.split(' ')
            classy = linelst[-1][:-1]
            if classy == '0':
                if count_ham < datanum1_ham:
                    file1.write(line)
                    count_ham += 1
                else:
                    file2.write(line)
            else:
                if count_spam < datanum1_spam:
                    file1.write(line)
                    count_spam += 1
                else:
                    file2.write(line)
    file1.close()
    file2.close()
    return 1



def score(X, Y, theta, wordpair, wpn):
    n = len(X[0])
    dummy = [0.0 for _ in xrange(n+wpn)]
    predict = np.array([0.0 for _ in xrange(len(Y))])

    for i, x in enumerate(X):
        wplist  = [0.0 for _ in xrange(wpn)]
        for p in range(len(wordpair)):
            if wordpair[p][0] in x and wordpair[p][1] in x:
                wplist[p] += 1
        dY_1 = np.array(list(x)+ wplist + dummy + [1.])
        dY_0 = np.array(dummy + list(x)+ wplist + [0.])
        predict[i] = 1.0 if (sum(dY_1 * theta) < sum(dY_0 * theta)) else 0.0

    correct = 0.

    # predict = np.array([0.0 for _ in xrange(len(Y))])

    for i, y in enumerate(Y):
        if predict[i] == y:
            correct += 1.

    print correct / len(Y)
    print predict, Y


if __name__ == '__main__':
    
    X, Y = read_data(2)
    wordpair = read_wordpair(WPN)
    theta = np.load('theta_2.14_addedge.npy')
    print score(X[0], Y[0], theta,wordpair, WPN)
    
    #a = segmentfile()

    # 0.888524590164

    # logreg = linear_model.LogisticRegression(C=1e5)
    # logreg.fit(X[1], Y[1])
    # print logreg.score(X[0], Y[0])
