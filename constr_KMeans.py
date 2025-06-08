import numpy as np
from sklearn.datasets import make_blobs

class ConKMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, seeds, X):
        if len(seeds) != self.n_clusters:
            print(seeds)
            print(self.n_clusters)
            raise ValueError("Seeds array and n_clusters must be the same size")
        seed_dict = {s:i for i, seed in enumerate(seeds) for s in seed}
        all_seeds = [s for row in seeds for s in row]
        all_seeds.sort()
        seed_assignment = [seed_dict[s] for s in all_seeds]
        seed_mask = np.zeros(len(X), dtype=bool)
        for s in all_seeds:
            seed_mask[s] = True
        # Initialize centroids how?
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)] 
        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X, seed_assignment, seed_mask)
             
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            #Generally won't happen, complex centroids
            #print(self.centroids)
            #print(new_centroids)
            #if np.all(self.centroids == new_centroids):
            #    break
                
            self.centroids = new_centroids

    def _assign_labels(self, X, seed_assignment=list(), seed_mask=list()):
        # Compute distances from each data point to centroids
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
         
        # Assign labels based on the nearest centroid
        assignments = np.argmin(distances, axis=1)
        assignments[seed_mask] = seed_assignment
        return assignments
    
    def _update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

