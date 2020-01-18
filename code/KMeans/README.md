# KMeans

**Type:** Clustering algorithm

**Parameter:** `dist` distance metric use (default euclidean), `seed` random seed, `max_iter` max iteration

**Usage:**
```python
# initialize KMeans algorithm as a new model
model = KMeans()

# Let model learn from train data, specify k cluster
model.fit(train_data, k)

# Display KMeans Report, contain:
# 1. visualize data point, centroids and voronoi graph (2d data only) --> from best k according to Silhouette
# 2. Elbow method and Silhouette graph
model.report()
```
See [this notebook](../notebook/KMeans_example.ipynb)

**Description:**

KMeans algorithm cluster data by trying to find the best centroid for each cluster that minimize distance variance from all sample to it's centroid. New sample data will be assign to cluster with nearest centroid.

KMeans pseudocode:
1. Choose k random data point as an initial centroids from sample data
2. Assign all sample data to the cluster with nearest centroid
3. Calculate mean from all sample each cluster, assign it as a new centroids
4. Repeat step 2 and 3 until centroids not change

Elbow method: 
Is the way to determine the best number of clusters for KMeans clustering, here the step:
1. Try KMeans with k from 2 - 10
2. For each k, calculate sse (sum distance from all sample to it's centroid)
3. Plot sse

Silhouette: 
Other way to determine the best number of clusters for KMeans clustering, more realible than elbow method, here the step:
1. Try KMeans with k from 2 - 10
2. For each k, calculate silhouette = (B - A)/max(A,B) with A is total distance from each data point to other point in same cluster, B is total distance from each data point to other point in the nearest cluster
3. Plot silhouette 

Voronoi:
Visualize cluster area on canvas, here the step:
1. Define canvas area, for example 200x200
2. For each point in canvas area, calculate the distance to each centroids
3. Assign each point to the nearest centroids