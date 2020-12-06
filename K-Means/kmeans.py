import numpy as np
from functions import *

class KMeans():
    def __init__(self, n_clusters=8, init='k-means++', n_init=5, max_iter=300):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.best_centers = None
        
    def fit(self, X):
        # try n_init times
        variance = np.inf
        for i in range(self.n_init):            
            C, labels = kmeans(X, self.n_clusters, self.init, self.max_iter)
            new_variance = np.linalg.norm(X - C[labels,:])
            if new_variance < variance:
                self.best_centers = C
                variance = new_variance
                
        return self
    
    def predict(self, X):
        labels = E_step(self.best_centers,X)
        
        return labels
            