import numpy as np


def generateData():
    return np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
                      (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
                      (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))


def getRandomCentroids(points, k):
    '''
        Select random k centroids from points
    '''
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def nearest_centroid(points, centroids):
    '''
        Returns an array containing the index to the nearest centroid for each point
    '''
    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def creates_new_centroids(points, nearest, centroids):
    '''
        Returns the new centroids assigned from the points closest to them
    '''
    return np.array([points[nearest == k].mean(axis=0) for k in range(centroids.shape[0])])


def k_means(points, k, num_iterations=100, e=0.00001):
    # Initialize centroids
    centroids = getRandomCentroids(points, k)

    # Run iterative process
    for i in range(num_iterations):
        nearest = nearest_centroid(points, centroids)
        new_centroids = creates_new_centroids(points, nearest, centroids)
        # checks whether the displacement of the point is less epsilon
        dif = abs(new_centroids - centroids)
        centroids = new_centroids
        if len(dif[dif > e]) == 0:
            break

    return centroids


def main():
    points = generateData()
    centroids = k_means(points, 3)
    print(centroids)


main()
