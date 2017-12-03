import numpy as np


def compute_error(weights, X, y):
    '''
        Computes Error = 1/N * (X * weights - y).T * (X * weights - y)
    '''
    N = len(X)
    diff = np.dot(X, weights) - y
    error = np.dot(diff.T, diff)

    return error[0][0] / N


def gradient_descent(weights, X, y, learning_rate, num_iterations):
    '''
        Performs gradient step num_iterations times
        in order to find optimal a, b values
    '''
    for i in range(num_iterations):
        weights = gradient_step(weights, X, y, learning_rate)

    return weights


def gradient_step(weights, X, y, learning_rate):
    '''
        Updates a and b in antigradient direction
        with given learning_rate
    '''
    N = len(X)
    # grad = (1 / N) * X.T * (X * weights - y)
    grad = np.dot(X.T, np.dot(X, weights) - y)
    return weights - (1. / N) * learning_rate * grad


def linear_regression(X, y, num_iterations=10000, learning_rate=0.0001):
    # add ones to X
    X = np.hstack((np.ones((len(X), 1)), X))
    # create weights
    weights = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
    weights = gradient_descent(weights, X, y, learning_rate, num_iterations)
    print(
        'End learning at weights = {0}, error = {1}'.format(
            weights.reshape(weights.shape[0]),
            compute_error(weights, X, y)
        )
    )

    return weights


def generateData():
    '''
    read data from 'data.csv'
    :return: X - points, y - predicted values
    '''
    # Step # 1 - Extract data
    points = np.genfromtxt('data.csv', delimiter=',')
    # Create X and Y
    N = len(points)
    X = points[:, 0].reshape(N, 1)
    y = points[:, 1].reshape(N, 1)
    return X, y


def main():
    X, y = generateData()
    weights = linear_regression(X, y)


main()
