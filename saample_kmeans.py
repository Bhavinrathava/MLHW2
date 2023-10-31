import numpy as np

class KMeans:
    def __init__(self, euclidean, cosim, n_clusters = 10, max_itr = 300, metric = "euclidean"):
        self.n_clusters = n_clusters
        self.max_itr = max_itr
        self.metric = metric
        self.euclidean = euclidean
        self.cosim = cosim

    def _compute_distance(self,a,b):
        if self.metric == 'euclidean':
            return self.euclidean(a, b)
        elif self.metric == 'cosim':
            return self.cosim(a, b)
        else:
            raise ValueError(f"Invalid metric {self.metric}. Supported metrics are 'euclidean' and 'cosim'.")


    
    def fit(self, x_train):
        # Randomly select centroids, uniformly distributed across the domain of the dataset
        min_, max_ = np.min(x_train, axis=0), np.max(x_train, axis=0)
        self.centroids = [np.random.uniform(min_, max_) for _ in range(self.n_clusters)]

        #Iterate, adjusting centroids untill converged or untill passed max_itr
        iteration = 0
        prev_centroids = None

        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_itr:

            #sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in x_train:
                dists = [self._compute_distance(x, centroid) for centroid in self.centroids]
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) if cluster else prev_centroids[i] for i, cluster in enumerate(sorted_points)]


            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]

            iteration += 1

    def predict(self, x_test):
        centroids = []
        centroid_idxs = []
        for x in x_test:
            dists = [self._compute_distance(x, centroid) for centroid in self.centroids]
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs 

