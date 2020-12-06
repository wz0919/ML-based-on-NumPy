import numpy as np
from functions import *

class GaussianMixture():
    def __init__(self, n_components=1, n_iter=100, n_init=1):
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_init = n_init
        
    def fit(self, X):
        best_nll = np.inf
        # run n_init times, choose paramters gives best nll
        for i in range(self.n_init):
            mu, sigma, pi = EM(X, self.n_components, self.n_iter)
            labels, nll = classify(pi, mu, sigma, X, calculating_nll = True)
            if nll < best_nll:
                best_nll = nll
                self.mu, self.sigma, self.pi = mu, sigma, pi
                
        return self
    
    def predict(self, X):
        labels, _ = classify(self.pi, self.mu, self.sigma, X)
        return labels
    
    

