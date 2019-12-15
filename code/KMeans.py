import numpy as np
import matplotlib.pyplot as plt

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
        return 'KMeans model with ' + str(self.k) + ' segment and distance ' + self.distance + '.'
    
    def init_centroids(self, d, k):
        index = np.random.choice(d.shape[0], k, replace=False)
        return d[index]
    
    def distance(self, point, target):
        if self.dist == 'euclidean':
            dist = np.sqrt(np.sum((point - target)**2))
        return dist

    def assign(self, data, centroids):
        dists = []
        for centroid in centroids:
            cur_dist = []
            for point in data:
                cur_dist.append(self.distance(point, centroid))
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
                dist += self.distance(point, centroid)
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
                        dist = int(self.distance(np.array([x,y]), np.array([x_cent,y_cent])))
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
        return ax
    
    def plot_elbow(self, sse_history, ax):
        sse_list = [sse_history[i][0] for i in range(1,len(sse_history)+1)]
        ax.plot(range(1,len(sse_list)+1), sse_list, 'ro-')
        return ax
    
    def silhouette(self, data, centroids, clusters):
        # TODO: create silhouette algorithm
        # while waiting, the best k from silhouette will always k=3
        if len(centroids) == 3:
            sil = 1
        else:
            sil = -1
        return sil

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
        for k in range(1, 11):
            centroids, clusters = self.fit(self.data, k)
            sse = self.sse(self.data, centroids, clusters)
            silhouette = self.silhouette(self.data, centroids, clusters)
            sse_history[k] = [sse, silhouette, centroids, clusters]
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        _, _, centroids, clusters = sse_history[best_k]
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        self.plot2d(self.data, centroids, clusters, ax[0])
        self.plot_elbow(sse_history, ax[1])