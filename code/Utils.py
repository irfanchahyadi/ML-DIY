import numpy as np

def distance(point, target, dist='euclidean'):
	if dist == 'euclidean':
	    dist = np.sqrt(np.sum((point - target)**2))
	return dist