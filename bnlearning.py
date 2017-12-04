import numpy as np
import math

"""     Hack:
Doc:    305
Words:  9314
Spam:   56
Ham:    249
"""
WORDS   = 9314
DOC     = 305
Spam    = 56
Ham     = 249


def inference(theta, data, n):
    """
    """
    Z = 0

    for d in data:
        # |d| = n + 1
        sum_theta_feature = 0

        Y = d[-1]

        if Y:
            for i in d[0]:
                # print i, type(i)
                sum_theta_feature += theta[i] * 1.0
        else:
            for i in d[0]:
                sum_theta_feature += theta[i + n] * 1.0

        sum_theta_feature += theta[-1] * d[-1]
        # print len(d[0]), sum_theta_feature
        Z += math.exp(sum_theta_feature)

    return Z




def get_first_term(n, data):
    first_term = np.array([0.0 for _ in xrange(2 * n + 1)])

    # for i in xrange(2 * n + 1):
    #     for d in data:
    #         first_term[i] += feature(d, i, n)
    #     first_term[i] /= n

    for d in data:
        if d[-1]:
            for index in d[0]:
                first_term[index] += 1.
        else:
            for index in d[0]:
                first_term[n + index] += 1.
        first_term[-1] += d[-1]

    first_term /= n


    # for i in xrange(n):
    #     for d in data:
    #         first_term[i] += d[i] and d[-1]
    #         first_term[i + n] += d[i] and !d[-1]
    #     first_term[i] /= len(n)
    #     first_term[i + n] /= len(n)
    #
    # for d in data:
    #     first_term[-1] += d[-1]
    # first_term[-1] /= len(n)

    return first_term


def feature(d, i, n):
    # if d[1] == 1:
    #     print "aaaaa"

    # print d[1]
    if i < n:
        # if i in d[0] and d[1] == 1:
        #     print "aaaaa", float(i in d[0] and d[1] == 1.)
        return float(i in d[0] and d[1] == 1.)
    elif i < 2 * n:
        # if i in d[0] and d[1] == 0.:
        #     print "aaaaa", float(i in d[0] and d[1] == 0.)
        return float(i - n in d[0] and d[1] == 0.)
    else:
        return float(d[1])


def get_second_term(theta, data, Z, n):
    second_term = np.array([0.0 for _ in xrange(2 * n + 1)])

    temps = np.array([0.0 for _ in xrange(2 * n + 1)])

    for index, d in enumerate(data):
        temp = 0.
        temp1 = 0.
        # ans, ans1 = [], []
        # print Y
        if d[1]:
            for i in d[0]:
                # ans.append(i)
                temp += theta[i] * 1.0
        else:
            for i in d[0]:
                # ans.append(i + n)
                temp += theta[i + n] * 1.0

        temp += theta[-1] * d[1]

        # for j in d[0]:
        #     if d[1]:
        #         temp += theta[j] * 1.0
        #     else:
        #         # print 'aaa'
        #         temp += theta[j + n] * 1.0
        #
        # temp += theta[-1] * d[1]

        # for j in xrange(2 * n + 1):
        #     if feature(d, j, n) == 1.:
        #         ans1.append(j)
        #     temp1 += theta[j] * feature(d, j, n)
        #
        # print temp, temp1
        # print sorted(ans)
        # print sorted(ans1)

        temps[index] = math.exp(temp)

    for index, d in enumerate(data):
        for i in d[0]:
            if d[1]:
                second_term[i] += 1.0 * temps[index]
            else:
                second_term[i + n] += 1.0 * temps[index]

            second_term[-1] += d[1]

    second_term /= Z

    # for i in xrange(2 * n + 1):
    #     for index, d in enumerate(data):
    #         second_term[i] += feature(d, i, n) * temps[index]
    #
    #     second_term[i] /= Z

    # for i in xrange(n):
    #     for d in data:
    #         temp = 0
    #         for j in xrange(n):
    #             temp += theta[j] * (d[j] and d[-1])
    #             temp += theta[j + n] * (d[j] and !d[-1])
    #         temp += theta[-1] * d[-1]
    #
    #         temp = exp(temp)
    #
    #         second_term[i] += (d[i] and d[-1]) * temp
    #         second_term[i + n] += (d[i] and !d[-1]) * temp
    #
    #     second_term[i] /= Z
    #     second_term[i + n] /= Z
    #
    # for d in data:
    #     temp = 0
    #     for j in xrange(n):
    #         temp += theta[j] * (d[j] and d[-1])
    #         temp += theta[j + n] * (d[j] and !d[-1])
    #     temp += theta[-1] * d[-1]
    #
    #     temp = exp(temp)
    #
    #     second_term[-1] += d[-1] * temp
    #
    # second_term[-1] /= Z

    return second_term


# def initialize_theta(n):


def read_data():

    data = []
    with open('data.txt', 'r') as datafile:
        for line in  datafile.readlines():
            line = line.split(' ')
            data.append((set(map(int, line[:-1])), int(line[-1])))
    return data


def main(alpha, delta):
    data = read_data()

    theta = np.random.rand(WORDS * 2 + 1)
    first_term = get_first_term(WORDS, data)
    print 'Got first_term ', str(first_term), sum(first_term)

    while True:
        Z = inference(theta, data, WORDS)
        print 'Inference Done, ', Z

        second_term = get_second_term(theta, data, Z, WORDS)
        print 'Got second_term'

        gredients = first_term - second_term

        print first_term
        print second_term
        print gredients

        theta += alpha * gredients

        print sum(map(abs, gredients))
        if abs(sum(gredients)) < delta:
            break

    return theta


if __name__ == '__main__':
    main(0.01, 0.1)