import sys
sys.path.append('../code/Utils')
import numpy as np
import matplotlib.pyplot as plt
from Utils import distance

class KNN():
    def __init__(self, dist='euclidean', seed=None):
        self.dist = dist
        np.random.seed(seed=seed)
    
    def __repr__(self):
        return 'KNN model distance ' + self.dist + '.'
    
    def fit(self, X, y, k):
        self.k = k
        self.known_X = X
        self.known_y = y

    def predict(self, X):
        res = []
        for point in X:
            dis = []
            for known_point in self.known_X:
                dis.append(distance(point, known_point, self.dist))
            res.append(self.known_y[np.argsort(dis)][:self.k].max())
        return np.array(res)
