import numpy as np
from scipy import stats


def generateData():
    num_observations = 300
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([-2, 3], [[2, .75], [.75, 2]], num_observations)
    X = np.vstack((x1, x2)).astype(np.float32)
    Y = np.hstack((np.zeros(num_observations),
                   np.ones(num_observations)))
    return X, Y


def splitDataset(X, Y, ratio=0.67):
    l = len(X)
    n_trn = int(l * ratio)
    ind = np.random.permutation(l)
    X = X[ind]
    Y = Y[ind]
    x_trn = X[:n_trn]
    y_trn = Y[:n_trn]
    x_tst = X[n_trn:]
    y_tst = Y[n_trn:]
    return x_trn, y_trn, x_tst, y_tst


def calc_all_distances(data_x, unknown):
    '''
        Function calculates distances between each pairs of known and unknown points
    '''
    return np.sqrt(((data_x - unknown[:, np.newaxis]) ** 2).sum(axis=2))


def predict(dists, data_y, k):
    '''
        Function predicts the class of the unknown point by the k nearest neighbours
    '''
    num_pred = dists.shape[0]
    y_pred = np.zeros(num_pred)
    for i in range(num_pred):
        dst = dists[i]
        y_nearest = data_y[np.argsort(dst)[:k]]
        y_pred[i] = stats.mode(y_nearest, axis=None).mode
    return y_pred


def accuracy(predicted, real):
    '''
        Calculates accuracy percentage
    '''
    correct = sum(predicted == real)
    l = len(predicted)
    return 100 * correct / l


def compare_k(data_x, data_y, test_x, test_y, kmin=1, kmax=50, kstep=4):
    k = list(range(kmin, kmax, kstep))
    steps = len(k)
    features = np.zeros(steps)

    print('Evaluating distancies started')

    distancies = calc_all_distances(data_x, test_x)
    miss = []
    results = []
    for j in range(steps):
        yk = predict(distancies, data_y, k[j])
        features[j] = accuracy(yk, test_y)
        results.append(yk)
        cond = yk != test_y
        miss.append({
            'k': k[j],
            'acc': features[j],
            'x': test_x[cond]}
        )

        print('k={0}, accuracy = {1}%'.format(k[j], features[j]))

    # find the best result
    ind_best = np.argmax(features)

    return results[ind_best]


def main():
    X, Y = generateData()
    x_trn, y_trn, x_tst, y_tst = splitDataset(X, Y)
    features = compare_k(x_trn, y_trn, x_tst, y_tst)
    print(features)

main()
