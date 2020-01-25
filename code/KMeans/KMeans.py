import sys
sys.path.append('../code/Utils')
import numpy as np
import matplotlib.pyplot as plt
from Utils import distance

class KMeans():
    def __init__(self, dist='euclidean', seed=None, max_iter=100):
        self.master_colormap = ['r', 'g', 'b', 'c', 'm', 'y', 'k', ]
        self.dist = dist
        self.centroids = None
        self.centroids_history = []
        self.clusters = None
        self.max_iter = max_iter
        self.last_iter = 0
        np.random.seed(seed=seed)
        
    def __repr__(self):
        return 'KMeans model with distance ' + self.dist + '.'
    
    def init_centroids(self, d, k):
        index = np.random.choice(d.shape[0], k, replace=False)
        return d[index]

    def assign(self, data, centroids):
        dists = []
        for centroid in centroids:
            cur_dist = []
            for point in data:
                cur_dist.append(distance(point, centroid, self.dist))
            dists.append(cur_dist)
        dists = np.array(dists).T
        return dists.argmin(axis=1)
    
    def update_centroids(self, data, centroids, clusters):
        new_centroids = []
        for cluster_id, centroid in enumerate(centroids):
            new_centroids.append(data[clusters == cluster_id].mean(axis=0))
        return np.array(new_centroids)
    
    def sse(self, data, centroids, clusters):
        dist = 0
        for cluster_id, centroid in enumerate(centroids):
            for point in data[clusters == cluster_id]:
                dist += distance(point, centroid, self.dist)
        return dist
    
    def plot2d(self, data, centroids, clusters, ax=None):
        x_max = 200
        y_max = 200
        
        def resize(point_before, range_before, range_after=(0,0,x_max,y_max), padding=0.1):
            shape_x = range_after[2] - range_after[0]
            shape_y = range_after[3] - range_after[1]
            pad_x = int(shape_x * padding)
            pad_y = int(shape_y * padding)
            range_inner = (range_after[0]+pad_x, range_after[1]+pad_x, range_after[2]-pad_y, range_after[3]-pad_y)
            new_point = [0,0]
            new_point[0] = ((point_before[0] - range_before[0])*(range_inner[2] - range_inner[0]))/(range_before[2] - range_before[0]) + range_inner[0]
            new_point[1] = ((point_before[1] - range_before[1])*(range_inner[3] - range_inner[1]))/(range_before[3] - range_before[1]) + range_inner[1]
            return new_point
        
        def voronoi(new_centroids, shape=(x_max,y_max)):
            depthmap = np.ones(shape, np.float)*1e308
            colormap = np.zeros(shape, np.int)

            for i, (x_cent,y_cent) in enumerate(new_centroids):
                paraboloid = []
                for x in range(shape[0]):
                    row = []
                    for y in range(shape[1]):
                        dist = int(distance(np.array([x,y]), np.array([x_cent,y_cent]), self.dist))
                        row.append(dist)
                    paraboloid.append(row)
                paraboloid = np.array(paraboloid)
                colormap = np.where(paraboloid < depthmap, i, colormap)
                depthmap = np.where(paraboloid < depthmap, paraboloid, depthmap)
            return colormap.T
    
        xi,yi = data.min(axis=0)
        xa,xa = data.max(axis=0)
        
        new_data = np.array([resize(i, (xi,yi,xa,xa)) for i in data])
        new_centroids = np.array([resize(i, (xi,yi,xa,xa)) for i in centroids])
        
        for cluster_id in range(len(new_centroids)):
            ax.scatter(new_data[:, 0][clusters == cluster_id], new_data[:, 1][clusters == cluster_id], color=self.colormap[cluster_id], alpha=0.2)
            ax.scatter(new_centroids[:, 0], new_centroids[:, 1], color=self.colormap[:len(new_centroids)])
        voronoi = voronoi(new_centroids)
        ax.imshow(voronoi, origin='lower', alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Best K=' + str(len(new_centroids)))
        return ax
    
    def plot_elbow(self, sse_history, ax):
        enum = range(2,len(sse_history)+1)
        sse_list = [sse_history[i][0] for i in enum]
        silh_list = [sse_history[i][1] for i in enum]
        ax.plot(enum, sse_list, 'ro-', label='Elbow')
        ax.plot([], [], 'go-', label='Silhouette', alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(enum, silh_list, 'go-', label='Silhouette', alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title('Elbow and Silhouette')
        return ax
    
    def silhouette(self, data, centroids, clusters):
        # Create index of nearest cluster for every cluster
        nearest = []
        for cluster_id_i, centroid_i in enumerate(centroids):
            dists = []
            for cluster_id_j, centroid_j in enumerate(centroids):
                if cluster_id_i == cluster_id_j:
                    dist = np.inf
                else:
                    dist = distance(centroid_i, centroid_j, self.dist)
                dists.append(dist)
            nearest.append(np.array(dists).argmin())
        
        silhouet = {}
        all_silhouet = 0
        for cluster_id, centroid in enumerate(centroids):
            silhouet[cluster_id] = []
            in_cluster = data[clusters == cluster_id]
            for point in in_cluster:
                distA = 0
                if len(in_cluster) > 1:
                    for other_point in in_cluster:
                        distA += distance(point, other_point, self.dist)
                    meanA = distA / (len(in_cluster) - 1)
                else:
                    meanA = 0

                distB = 0
                nearest_cluster = data[clusters == nearest[cluster_id]]
                if len(nearest_cluster) > 0:
                    for nearest_cluster_point in nearest_cluster:
                        distB += distance(point, nearest_cluster_point, self.dist)
                    meanB = distB / (len(nearest_cluster))
                else:
                    meanB = 0
                silh = (meanB - meanA)/max(meanA, meanB)
                silhouet[cluster_id].append(silh)
                all_silhouet += silh
        mean_silhouet = all_silhouet/data.shape[0]
        return mean_silhouet

    def fit(self, data, k):
        self.k = k
        self.colormap = (self.master_colormap * int(np.ceil(k/7)))[:k]
        self.data = data
        centroids = self.init_centroids(self.data, self.k)
        self.centroids_history.append(centroids)
        for i in range(self.max_iter):
            clusters = self.assign(self.data, centroids)
            centroids = self.update_centroids(self.data, centroids, clusters)
            self.centroids_history.append(centroids)
            if np.array_equal(self.centroids_history[-2], centroids):
                self.last_iter = i + 1
                break
        return centroids, clusters
    
    def report(self):
        sse_history = {}
        best_silhouette = -1
        best_k = None
        for k in range(2, 11):
            centroids, clusters = self.fit(self.data, k)
            sse = self.sse(self.data, centroids, clusters)
            silhouette = self.silhouette(self.data, centroids, clusters)
            sse_history[k] = [sse, silhouette, centroids, clusters]
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        _, _, self.centroids, clusters = sse_history[best_k]
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        self.plot2d(self.data, self.centroids, clusters, ax[0])
        self.plot_elbow(sse_history, ax[1])

    def predict(self, data):
        return self.assign(data, self.centroids)