import numpy as np
from sklearn.datasets import make_blobs
import torch
import utils
class ConKMeans:
    def __init__(self, n_clusters, sim_metric, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.sim_metric = sim_metric

    def fit(self, X, Y, senses, seeds=None):
        if seeds:
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
            centr_ids = [x[0:n_seeds] for x in seeds]
        #mean n_seeds for each cluster
            self.centroids = torch.stack([torch.mean(torch.stack(list(train.iloc[seed_ids][emb_name])), axis=1) for seed_ids in centr_ids])

        else:
            self.centroids = X[torch.randperm(X.size(0))[:self.n_clusters]]
            seed_assignment = list()
            seed_mask = list()
        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            _, labels = self._assign_labels(X, seed_assignment, seed_mask)
             
            # Update centroids
            new_centroids = self._update_centroids(X, labels) 
               
            self.centroids = new_centroids
        return utils.agirre_matr(labels, Y, senses)

    def _assign_labels(self, X, seed_assignment=list(), seed_mask=list()):
        # Compute distances from each data point to centroids
        
        distances = self.sim_metric(X, self.centroids)
        # Assign labels based on the nearest centroid
        assignments = np.argmin(distances, axis=1)
        assignments[seed_mask] = seed_assignment
        return distances, assignments
    
    def _update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

